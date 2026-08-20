"""Differentiable simulation and information diagnostics for evaporating black holes.

Research status
---------------
This module is a reference implementation for the two *stopped effective models*
described in ``black_hole_information_blueprint.md``.  It is not a claim that either
model remains valid at Planck mass.  Every path is stopped at ``m_floor > 0``.

The module deliberately separates two derivatives that are often conflated:

1. ``pathwise`` derivatives propagate through reparameterized simulation noise;
2. ``score`` derivatives hold the realized data fixed and differentiate the path
   log likelihood.  The latter, not the former, defines the Fisher information.

The jump simulator has an unavoidable discrete topology (event count and terminal
event).  Inverse-transform sampling gives an almost-everywhere pathwise derivative
conditional on that topology.  It does *not* by itself give an unbiased derivative
through changes in event count.  Likelihood-ratio or weak-derivative corrections are
needed when that distinction matters.

Requires PyTorch >= 2.2.  All calculations default to float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
LOG_2PI = torch.log(torch.tensor(2.0 * pi, dtype=torch.float64))


def _scalar(value: float, like: Tensor) -> Tensor:
    """Create a scalar on the same device and with the same dtype as ``like``."""

    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _normal_log_prob(value: Tensor, mean: Tensor, sd: Tensor) -> Tensor:
    """Scalar Gaussian log density with explicit constants."""

    log_2pi = LOG_2PI.to(dtype=value.dtype, device=value.device)
    z = (value - mean) / sd
    return -0.5 * (log_2pi + 2.0 * torch.log(sd) + z.square())


def positive_parameters(raw_theta: Tensor, m_floor: float) -> tuple[Tensor, Tensor]:
    """Map an unconstrained vector to ``theta=(M0, gamma, alpha)``.

    Returns the physical parameter vector and the log absolute Jacobian.  This is
    the recommended parameterization for HMC/NUTS because the sampler never proposes
    a non-positive mass or rate.  The Jacobian must be added to a density written on
    the physical parameter scale.
    """

    if raw_theta.shape != (3,):
        raise ValueError("raw_theta must have shape (3,) ordered as M0, gamma, alpha")
    floor = _scalar(m_floor, raw_theta)
    physical = torch.stack(
        (
            floor + F.softplus(raw_theta[0]),
            F.softplus(raw_theta[1]),
            F.softplus(raw_theta[2]),
        )
    )
    log_abs_det_jacobian = F.logsigmoid(raw_theta).sum()
    return physical, log_abs_det_jacobian


@dataclass(frozen=True)
class PhysicsConfig:
    """Numerical and physical constants in Planck units."""

    horizon: float = 1.0
    dt: float = 1.0e-3
    m_floor: float = 0.25
    hawking_kappa: float = 1.0 / (8.0 * pi)  # T_H(M) = kappa / M
    sigma_base: float = 0.02
    initial_observation_sd: float = 0.05
    detector_efficiency: float = 1.0
    observation_gain: float = 1.0
    observation_sd: float = 0.10
    max_events: int = 10000
    planck_mixture_terms: int = 256
    eps: float = 1.0e-12

    def validate(self) -> None:
        if self.horizon <= 0.0 or self.dt <= 0.0:
            raise ValueError("horizon and dt must be positive")
        if self.m_floor <= 0.0:
            raise ValueError("m_floor must be strictly positive")
        if self.sigma_base <= 0.0 or self.observation_sd <= 0.0:
            raise ValueError("diffusion and observation scales must be positive")
        if self.initial_observation_sd <= 0.0:
            raise ValueError("initial_observation_sd must be positive")
        if not 0.0 < self.detector_efficiency <= 1.0:
            raise ValueError("detector_efficiency must be in (0, 1]")
        if self.max_events < 1 or self.planck_mixture_terms < 2:
            raise ValueError("max_events and planck_mixture_terms are too small")


@dataclass
class JumpPath:
    """One stopped marked point-process realization."""

    event_times: Tensor
    energies: Tensor
    terminal: Tensor
    final_mass: Tensor
    absorbed: bool
    horizon: float

    def detached(self) -> JumpPath:
        return JumpPath(
            event_times=self.event_times.detach(),
            energies=self.energies.detach(),
            terminal=self.terminal.detach(),
            final_mass=self.final_mass.detach(),
            absorbed=self.absorbed,
            horizon=self.horizon,
        )


@dataclass
class DiffusionPath:
    """One Euler-Maruyama path with an absorbing cutoff.

    ``regular_transition[k]`` is true when transition k is represented by an
    ordinary Gaussian Euler density.  ``hit_transition[k]`` identifies the first
    censored transition into the absorbing boundary.  Later increments contribute
    no likelihood.
    """

    times: Tensor
    masses: Tensor
    regular_transition: Tensor
    hit_transition: Tensor

    def detached(self) -> DiffusionPath:
        return DiffusionPath(
            times=self.times.detach(),
            masses=self.masses.detach(),
            regular_transition=self.regular_transition.detach(),
            hit_transition=self.hit_transition.detach(),
        )


class PlanckNumberMarks:
    r"""Finite-mixture representation of the dimensionless Planck number law.

    The target density is proportional to ``x^2 / (exp(x)-1)``.  Expanding the
    denominator gives a mixture

        K ~ p_k proportional to k^{-3},   X | K=k ~ Gamma(shape=3, rate=k).

    Truncating the mixture at ``k_max`` is exponentially accurate away from zero
    and, importantly, makes sampling and likelihood evaluation internally
    consistent.  Because shape=3 is integer, Gamma sampling is implemented as a
    sum of three exponential variables and needs no non-PyTorch dependency.
    """

    def __init__(self, k_max: int = 256, eps: float = 1.0e-12):
        self.k_max = int(k_max)
        self.eps = float(eps)

    def _components(self, like: Tensor) -> tuple[Tensor, Tensor]:
        k = torch.arange(1, self.k_max + 1, dtype=like.dtype, device=like.device)
        log_weights = -3.0 * torch.log(k)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
        return k, log_weights

    def sample_dimensionless(self, like: Tensor, generator: torch.Generator) -> Tensor:
        k, log_weights = self._components(like)
        index = torch.multinomial(log_weights.exp(), 1, generator=generator)
        rate = k[index].squeeze(0)
        u = torch.rand((3,), dtype=like.dtype, device=like.device, generator=generator)
        u = u.clamp_min(self.eps)
        return -torch.log(u).sum() / rate

    def log_prob_dimensionless(self, x: Tensor) -> Tensor:
        k, log_weights = self._components(x)
        safe_x = x.clamp_min(self.eps)
        # Gamma(shape=3, rate=k): k^3 x^2 exp(-kx) / Gamma(3), Gamma(3)=2.
        component_log_pdf = (
            3.0 * torch.log(k)
            + 2.0 * torch.log(safe_x)
            - k * safe_x
            - torch.log(_scalar(2.0, x))
        )
        return torch.logsumexp(log_weights + component_log_pdf, dim=0)

    def log_survival_dimensionless(self, x: Tensor) -> Tensor:
        """Log P(X >= x); used for the terminal boundary atom."""

        k, log_weights = self._components(x)
        z = k * x.clamp_min(0.0)
        log_survival_component = -z + torch.log1p(z + 0.5 * z.square())
        return torch.logsumexp(log_weights + log_survival_component, dim=0)

    def moment(self, order: int, like: Tensor) -> Tensor:
        """Return E[X**order] for a nonnegative integer order."""

        if order < 0 or int(order) != order:
            raise ValueError("order must be a nonnegative integer")
        k, log_weights = self._components(like)
        order_t = _scalar(float(order), like)
        log_gamma_ratio = torch.lgamma(_scalar(3.0 + order, like)) - torch.lgamma(
            _scalar(3.0, like)
        )
        component_log_moment = log_gamma_ratio - order_t * torch.log(k)
        return torch.sum(log_weights.exp() * component_log_moment.exp())


class PacketJumpDiffusion:
    """Paradigm A: stopped non-homogeneous marked point process."""

    def __init__(self, config: PhysicsConfig):
        config.validate()
        self.config = config
        self.marks = PlanckNumberMarks(config.planck_mixture_terms, config.eps)

    def intensity(self, mass: Tensor, gamma: Tensor) -> Tensor:
        return gamma / mass.clamp_min(self.config.m_floor)

    def temperature(self, mass: Tensor) -> Tensor:
        return _scalar(self.config.hawking_kappa, mass) / mass

    def simulate(self, theta: Tensor, generator: torch.Generator) -> JumpPath:
        """Simulate by time rescaling and reparameterized Planck marks.

        Branch decisions are made from detached values.  Gradients therefore flow
        through waiting times and jump sizes conditional on the realized topology.
        """

        m0, gamma, _alpha_unused = theta.unbind()
        if float(m0.detach()) <= self.config.m_floor:
            raise ValueError("M0 must exceed m_floor")

        t = torch.zeros((), dtype=theta.dtype, device=theta.device)
        mass = m0
        event_times: list[Tensor] = []
        energies: list[Tensor] = []
        terminal_flags: list[bool] = []
        absorbed = False

        for _ in range(self.config.max_events):
            rate = self.intensity(mass, gamma)
            u = torch.rand((), dtype=theta.dtype, device=theta.device, generator=generator)
            wait = -torch.log(u.clamp_min(self.config.eps)) / rate
            next_t = t + wait
            if float(next_t.detach()) >= self.config.horizon:
                break

            x = self.marks.sample_dimensionless(theta, generator)
            proposed_energy = self.temperature(mass) * x
            available = mass - _scalar(self.config.m_floor, mass)
            terminal = bool((proposed_energy >= available).detach().item())
            energy = torch.minimum(proposed_energy, available)

            event_times.append(next_t)
            energies.append(energy)
            terminal_flags.append(terminal)
            t = next_t

            if terminal:
                mass = _scalar(self.config.m_floor, mass)
                absorbed = True
                break
            mass = mass - energy
        else:
            raise RuntimeError(
                "max_events reached; increase the cap or move m_floor away from the singular regime"
            )

        empty = torch.empty((0,), dtype=theta.dtype, device=theta.device)
        return JumpPath(
            event_times=torch.stack(event_times) if event_times else empty,
            energies=torch.stack(energies) if energies else empty,
            terminal=torch.tensor(terminal_flags, dtype=torch.bool, device=theta.device),
            final_mass=mass,
            absorbed=absorbed,
            horizon=self.config.horizon,
        )

    def complete_path_loglik(self, path: JumpPath, theta: Tensor, y0: Tensor) -> Tensor:
        """Complete-data log likelihood for a fixed realized marked path.

        The compensator is exact because the mass, hence the intensity, is constant
        between jumps.  A jump that reaches ``m_floor`` is treated as a terminal
        boundary atom whose probability is the Planck tail beyond the available
        energy.  The terminal packet's exact clamped energy is not treated as a
        continuous observation.
        """

        m0, gamma, _alpha_unused = theta.unbind()
        sd0 = _scalar(self.config.initial_observation_sd, theta)
        ll = _normal_log_prob(y0, m0, sd0)
        mass = m0
        previous_t = torch.zeros((), dtype=theta.dtype, device=theta.device)

        for i in range(path.event_times.numel()):
            event_t = path.event_times[i]
            exposure = event_t - previous_t
            rate = self.intensity(mass, gamma)
            ll = ll + torch.log(rate) - rate * exposure

            temp = self.temperature(mass)
            if bool(path.terminal[i].item()):
                x_max = (mass - _scalar(self.config.m_floor, mass)) / temp
                ll = ll + self.marks.log_survival_dimensionless(x_max)
                mass = _scalar(self.config.m_floor, mass)
            else:
                energy = path.energies[i]
                x = energy / temp
                ll = ll + self.marks.log_prob_dimensionless(x) - torch.log(temp)
                mass = mass - energy
            previous_t = event_t

        if not path.absorbed:
            final_exposure = _scalar(path.horizon, theta) - previous_t
            ll = ll - self.intensity(mass, gamma) * final_exposure
        return ll

    def mass_on_grid(self, path: JumpPath, m0: Tensor, grid: Tensor) -> Tensor:
        """Evaluate the càdlàg mass path at fixed grid times."""

        mass = m0
        masses: list[Tensor] = []
        event_index = 0
        for time in grid:
            while (
                event_index < path.event_times.numel()
                and bool((path.event_times[event_index] <= time).detach().item())
            ):
                if bool(path.terminal[event_index].item()):
                    mass = _scalar(self.config.m_floor, mass)
                else:
                    mass = mass - path.energies[event_index]
                event_index += 1
            masses.append(mass)
        return torch.stack(masses)

    def entropy_drift(self, mass: Tensor, gamma: Tensor) -> Tensor:
        r"""Interior generator applied to S_BH(M)=4*pi*M^2.

        This expression uses the untruncated Planck moments and is therefore an
        interior approximation.  The exact stopped generator replaces the moments
        by an integral with the terminal boundary atom.
        """

        c1 = self.marks.moment(1, mass)
        c2 = self.marks.moment(2, mass)
        temp = self.temperature(mass)
        rate = self.intensity(mass, gamma)
        return rate * (-8.0 * pi * mass * temp * c1 + 4.0 * pi * temp.square() * c2)


class ContinuousMassDiffusion:
    """Paradigm B: stopped Itô diffusion integrated by Euler-Maruyama."""

    def __init__(self, config: PhysicsConfig, moment_matched: bool = False):
        config.validate()
        self.config = config
        self.moment_matched = bool(moment_matched)
        self._marks = PlanckNumberMarks(config.planck_mixture_terms, config.eps)

    def drift(self, mass: Tensor, alpha: Tensor) -> Tensor:
        return -alpha / mass.square()

    def diffusion(self, mass: Tensor, gamma: Tensor | None = None) -> Tensor:
        if not self.moment_matched:
            return _scalar(self.config.sigma_base, mass) / mass
        if gamma is None:
            raise ValueError("gamma is required for the moment-matched diffusion")
        c2 = self._marks.moment(2, mass)
        coefficient = torch.sqrt(gamma * _scalar(self.config.hawking_kappa**2, mass) * c2)
        return coefficient / mass.pow(1.5)

    def simulate(self, theta: Tensor, generator: torch.Generator) -> DiffusionPath:
        m0, gamma, alpha = theta.unbind()
        n_steps = int(round(self.config.horizon / self.config.dt))
        if n_steps < 1:
            raise ValueError("horizon/dt must define at least one Euler step")
        dt = _scalar(self.config.horizon / n_steps, theta)
        sqrt_dt = torch.sqrt(dt)
        times = torch.linspace(
            0.0,
            self.config.horizon,
            n_steps + 1,
            dtype=theta.dtype,
            device=theta.device,
        )

        mass = m0
        masses: list[Tensor] = [mass]
        regular: list[bool] = []
        hit: list[bool] = []
        alive = True
        floor = _scalar(self.config.m_floor, theta)

        for _ in range(n_steps):
            if not alive:
                masses.append(floor)
                regular.append(False)
                hit.append(False)
                continue
            noise = torch.randn((), dtype=theta.dtype, device=theta.device, generator=generator)
            candidate = mass + self.drift(mass, alpha) * dt
            candidate = candidate + self.diffusion(mass, gamma) * sqrt_dt * noise
            boundary_hit = bool((candidate <= floor).detach().item())
            if boundary_hit:
                mass = floor
                alive = False
                regular.append(False)
                hit.append(True)
            else:
                mass = candidate
                regular.append(True)
                hit.append(False)
            masses.append(mass)

        return DiffusionPath(
            times=times,
            masses=torch.stack(masses),
            regular_transition=torch.tensor(regular, dtype=torch.bool, device=theta.device),
            hit_transition=torch.tensor(hit, dtype=torch.bool, device=theta.device),
        )

    def girsanov_loglik(self, path: DiffusionPath, theta: Tensor, y0: Tensor) -> Tensor:
        r"""Stopped Euler approximation to the Girsanov log-likelihood ratio.

        The reference measure has zero drift and the same diffusion coefficient.
        A regular increment contributes

            (b/a) dM - 0.5 (b^2/a) dt,   a=sigma^2.

        The boundary-hitting increment is censored and contributes the log ratio of
        Gaussian hitting probabilities under the target and reference drifts.
        """

        m0, gamma, alpha = theta.unbind()
        sd0 = _scalar(self.config.initial_observation_sd, theta)
        ll = _normal_log_prob(y0, m0, sd0)
        floor = _scalar(self.config.m_floor, theta)
        eps = _scalar(self.config.eps, theta)

        for k in range(path.times.numel() - 1):
            if not bool((path.regular_transition[k] | path.hit_transition[k]).item()):
                continue
            mass = path.masses[k]
            dt = path.times[k + 1] - path.times[k]
            b = self.drift(mass, alpha)
            s = self.diffusion(mass, gamma)
            a = s.square()
            if bool(path.regular_transition[k].item()):
                dm = path.masses[k + 1] - mass
                ll = ll + (b / a) * dm - 0.5 * (b.square() / a) * dt
            else:
                scale = s * torch.sqrt(dt)
                z_target = (floor - mass - b * dt) / scale
                z_reference = (floor - mass) / scale
                p_target = torch.special.ndtr(z_target).clamp_min(eps)
                p_reference = torch.special.ndtr(z_reference).clamp_min(eps)
                ll = ll + torch.log(p_target) - torch.log(p_reference)
        return ll

    def entropy_drift(self, mass: Tensor, gamma: Tensor, alpha: Tensor) -> Tensor:
        r"""Itô drift of S_BH(M)=4*pi*M^2."""

        s = self.diffusion(mass, gamma)
        return 8.0 * pi * mass * self.drift(mass, alpha) + 4.0 * pi * s.square()


def detector_signal(mass: Tensor, gain: float) -> Tensor:
    """Continuous detector signal h(M)=gain/M^2."""

    return _scalar(gain, mass) / mass.square()


def point_process_information(
    full_rate: Tensor,
    reduced_rate: Tensor,
    dt: float,
    eps: float = 1.0e-12,
) -> Tensor:
    r"""Expected counting-process information in nats.

    For predictable intensities lambda* and lambda_bar, the integrand is

        lambda* log(lambda*/lambda_bar) - lambda* + lambda_bar.

    It is transfer entropy only when ``reduced_rate`` is the observation-history
    filter E[lambda* | F^Y_{t-}].  Any other baseline produces a valid likelihood
    information contrast but must not be labeled transfer entropy.
    """

    full = full_rate.clamp_min(eps)
    reduced = reduced_rate.clamp_min(eps)
    return _scalar(dt, full) * (full * torch.log(full / reduced) - full + reduced).sum(dim=-1)


def diffusion_information(
    full_signal: Tensor,
    reduced_signal: Tensor,
    observation_sd: float,
    dt: float,
) -> Tensor:
    r"""Expected Gaussian-channel information in nats.

    For dY=h(M)dt+rho*dV, the conditional information rate is
    0.5*(h-h_hat)^2/rho^2.  It is transfer entropy when h_hat is the
    F^Y_t-predictable filtered signal.
    """

    rho2 = _scalar(observation_sd**2, full_signal)
    return 0.5 * _scalar(dt, full_signal) * (
        (full_signal - reduced_signal).square() / rho2
    ).sum(dim=-1)


def score_vector(log_likelihood: Tensor, theta: Tensor) -> Tensor:
    """Differentiate a fixed-data log likelihood with respect to theta."""

    (score,) = torch.autograd.grad(
        log_likelihood,
        theta,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    return score


def empirical_fisher(scores: Tensor) -> Tensor:
    """Outer-product-of-scores Monte Carlo estimator."""

    if scores.ndim != 2 or scores.shape[1] != 3:
        raise ValueError("scores must have shape (replicates, 3)")
    # Fisher information is E[s_theta s_theta^T], not the finite-sample covariance
    # of the score.  The population mean score is zero under regularity conditions,
    # but centering a small Monte Carlo sample can erase genuine information (for
    # example, when every rare-event path happens to contain zero events).
    return scores.transpose(0, 1) @ scores / scores.shape[0]


def common_parameter_pullback(
    fisher_a: Tensor,
    fisher_b: Tensor,
    hawking_kappa: float,
    planck_mean_x: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Compare both models on psi=(M0, alpha_eff).

    Matching the packet mean drift to -alpha_eff/M^2 gives

        alpha_eff = gamma * hawking_kappa * E[X].

    The native three-parameter Fisher matrices are structurally singular because
    alpha is inactive in A and gamma is inactive in the requested B.  Pullback by
    the two Jacobians before differencing.
    """

    dtype, device = fisher_a.dtype, fisher_a.device
    zero = torch.zeros((), dtype=dtype, device=device)
    one = torch.ones((), dtype=dtype, device=device)
    scale = one / (_scalar(hawking_kappa, fisher_a) * planck_mean_x)
    jacobian_a = torch.stack(
        (torch.stack((one, zero)), torch.stack((zero, scale)), torch.stack((zero, zero)))
    )
    jacobian_b = torch.stack(
        (torch.stack((one, zero)), torch.stack((zero, zero)), torch.stack((zero, one)))
    )
    common_a = jacobian_a.transpose(0, 1) @ fisher_a @ jacobian_a
    common_b = jacobian_b.transpose(0, 1) @ fisher_b @ jacobian_b
    return common_a, common_b, common_b - common_a


def _leave_one_out_mean(values: Tensor) -> Tensor:
    """Open-loop ensemble baseline used only when no causal filter is supplied."""

    if values.shape[0] < 2:
        raise ValueError("at least two replicates are required for a leave-one-out baseline")
    return (values.sum(dim=0, keepdim=True) - values) / (values.shape[0] - 1)


class CentralizedInformationEngine:
    """Parallel Monte Carlo evaluator for both paradigms.

    ``evaluate`` returns raw and common-parameter Fisher matrices, integrated
    point-process/Gaussian-channel information, entropy drifts, and the pathwise
    sensitivity of the information contrast.  Supply observation-history filtered
    predictors to obtain transfer entropy.  If they are omitted, the method uses a
    leave-one-out ensemble marginal and labels the result as an open-loop contrast.
    """

    def __init__(self, config: PhysicsConfig):
        self.config = config
        self.jump = PacketJumpDiffusion(config)
        self.diffusion = ContinuousMassDiffusion(config, moment_matched=False)
        self.diffusion_matched = ContinuousMassDiffusion(config, moment_matched=True)

    def evaluate(
        self,
        theta: Tensor,
        replicates: int,
        seed: int = 20260820,
        reduced_rate_a: Tensor | None = None,
        reduced_signal_b: Tensor | None = None,
    ) -> dict[str, Tensor | str]:
        if theta.shape != (3,) or not theta.requires_grad:
            raise ValueError("theta must be a length-3 leaf tensor with requires_grad=True")
        if replicates < 2:
            raise ValueError("replicates must be at least 2")

        n_steps = int(round(self.config.horizon / self.config.dt))
        dt = self.config.horizon / n_steps
        grid = torch.linspace(
            0.0,
            self.config.horizon,
            n_steps + 1,
            dtype=theta.dtype,
            device=theta.device,
        )
        generator = torch.Generator(device=theta.device)
        generator.manual_seed(seed)

        jump_masses: list[Tensor] = []
        diffusion_masses: list[Tensor] = []
        scores_a: list[Tensor] = []
        scores_b: list[Tensor] = []

        for _ in range(replicates):
            path_a = self.jump.simulate(theta, generator)
            path_b = self.diffusion.simulate(theta, generator)

            # A shared noisy initial measurement makes M0 a dominated parameter in
            # both likelihoods.  Without it, a deterministic initial condition can
            # give a singular or nonregular Fisher problem.
            z0 = torch.randn((), dtype=theta.dtype, device=theta.device, generator=generator)
            y0 = theta[0] + _scalar(self.config.initial_observation_sd, theta) * z0

            # IMPORTANT: Fisher scores differentiate a likelihood of fixed data.
            # Detaching the simulated realization prevents contamination by the
            # pathwise derivative of the data-generating map.
            ll_a = self.jump.complete_path_loglik(path_a.detached(), theta, y0.detach())
            ll_b = self.diffusion.girsanov_loglik(path_b.detached(), theta, y0.detach())
            scores_a.append(score_vector(ll_a, theta))
            scores_b.append(score_vector(ll_b, theta))

            jump_masses.append(self.jump.mass_on_grid(path_a, theta[0], grid))
            diffusion_masses.append(path_b.masses)

        mass_a = torch.stack(jump_masses)
        mass_b = torch.stack(diffusion_masses)
        score_a = torch.stack(scores_a)
        score_b = torch.stack(scores_b)
        fisher_a = empirical_fisher(score_a)
        fisher_b = empirical_fisher(score_b)

        full_rate_a = (
            _scalar(self.config.detector_efficiency, theta)
            * theta[1]
            / mass_a.clamp_min(self.config.m_floor)
        )
        full_signal_b = detector_signal(mass_b, self.config.observation_gain)

        if reduced_rate_a is None:
            reduced_rate_a = _leave_one_out_mean(full_rate_a)
            rate_label = "open_loop_leave_one_out; not transfer entropy"
        else:
            if reduced_rate_a.shape != full_rate_a.shape:
                raise ValueError("reduced_rate_a has the wrong shape")
            rate_label = "user_supplied_predictable_filter"

        if reduced_signal_b is None:
            reduced_signal_b = _leave_one_out_mean(full_signal_b)
            signal_label = "open_loop_leave_one_out; not transfer entropy"
        else:
            if reduced_signal_b.shape != full_signal_b.shape:
                raise ValueError("reduced_signal_b has the wrong shape")
            signal_label = "user_supplied_predictable_filter"

        info_a_by_path = point_process_information(
            full_rate_a, reduced_rate_a, dt, self.config.eps
        )
        info_b_by_path = diffusion_information(
            full_signal_b, reduced_signal_b, self.config.observation_sd, dt
        )
        info_a = info_a_by_path.mean()
        info_b = info_b_by_path.mean()
        delta_information = info_b - info_a

        # Pathwise sensitivity is conditional on the jump event topology.  It is
        # reported separately and is never substituted for a likelihood score.
        (delta_pathwise_gradient,) = torch.autograd.grad(
            delta_information,
            theta,
            retain_graph=False,
            create_graph=False,
        )

        c1 = self.jump.marks.moment(1, theta)
        common_a, common_b, delta_fisher = common_parameter_pullback(
            fisher_a, fisher_b, self.config.hawking_kappa, c1
        )

        entropy_rate_a = self.jump.entropy_drift(mass_a, theta[1]).mean()
        entropy_rate_b = self.diffusion.entropy_drift(mass_b, theta[1], theta[2]).mean()

        return {
            "fisher_a_native": fisher_a,
            "fisher_b_native": fisher_b,
            "fisher_a_common_M0_alpha_eff": common_a,
            "fisher_b_common_M0_alpha_eff": common_b,
            "delta_fisher_common_B_minus_A": delta_fisher,
            "information_a_nats": info_a.detach(),
            "information_b_nats": info_b.detach(),
            "delta_information_B_minus_A_nats": delta_information.detach(),
            "delta_information_pathwise_gradient": delta_pathwise_gradient.detach(),
            "mean_entropy_rate_a": entropy_rate_a.detach(),
            "mean_entropy_rate_b": entropy_rate_b.detach(),
            "a_predictor_status": rate_label,
            "b_predictor_status": signal_label,
            "gradient_semantics": (
                "Fisher uses fixed-data likelihood scores; information gradient is "
                "pathwise and conditional on the realized jump topology"
            ),
        }


def mean_drift_matching(config: PhysicsConfig, like: Tensor) -> dict[str, Tensor]:
    """Return the exact finite-mixture moment-matching relationships.

    For the packet model,

        E[dM | M]/dt = -(gamma*kappa*E[X])/M^2,
        Var[dM | M]/dt = gamma*kappa^2*E[X^2]/M^3.

    Therefore the requested sigma_base/M diffusion does not match the packet
    quadratic variation globally.  The matching diffusion scales as M^{-3/2}.
    """

    marks = PlanckNumberMarks(config.planck_mixture_terms, config.eps)
    c1 = marks.moment(1, like)
    c2 = marks.moment(2, like)
    kappa = _scalar(config.hawking_kappa, like)
    return {
        "planck_mean_x": c1,
        "planck_second_moment_x": c2,
        "alpha_per_gamma": kappa * c1,
        "matched_diffusion_prefactor_per_sqrt_gamma": kappa * torch.sqrt(c2),
    }


__all__ = [
    "CentralizedInformationEngine",
    "ContinuousMassDiffusion",
    "DiffusionPath",
    "JumpPath",
    "PacketJumpDiffusion",
    "PhysicsConfig",
    "PlanckNumberMarks",
    "common_parameter_pullback",
    "diffusion_information",
    "empirical_fisher",
    "mean_drift_matching",
    "point_process_information",
    "positive_parameters",
    "score_vector",
]
