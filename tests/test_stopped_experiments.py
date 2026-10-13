"""Independent numerical oracles for stopped-model and likelihood contracts."""

import math
import unittest
from unittest import mock

import numpy as np
import torch

from black_hole_information.engine import (
    CentralizedInformationEngine,
    ContinuousMassDiffusion,
    DiffusionPath,
    JumpPath,
    PacketJumpDiffusion,
    PhysicsConfig,
    PlanckNumberMarks,
    _leave_one_out_mean,
    mean_drift_matching,
    point_process_information,
    positive_parameters,
)

torch.set_default_dtype(torch.float64)


def one_step_path(theta, config, *, hit=False):
    floor = torch.tensor(config.m_floor)
    increment = -theta[2] / theta[0].square() * config.horizon
    return DiffusionPath(
        times=torch.tensor([0.0, config.horizon]),
        masses=torch.stack((theta[0], floor if hit else theta[0] + increment)),
        regular_transition=torch.tensor([not hit]),
        hit_transition=torch.tensor([hit]),
    )


class StoppingTests(unittest.TestCase):
    def test_absorbed_packet_intensity_is_zero(self):
        model = PacketJumpDiffusion(PhysicsConfig())
        mass = torch.tensor([0.25, 1.0])
        torch.testing.assert_close(
            model.intensity(mass, torch.tensor(4.0)), torch.tensor([0.0, 4.0])
        )

    def test_both_entropy_generators_vanish_at_the_cutoff(self):
        config = PhysicsConfig()
        mass, gamma, alpha = torch.tensor(0.25), torch.tensor(4.0), torch.tensor(1.0)
        self.assertEqual(float(PacketJumpDiffusion(config).entropy_drift(mass, gamma)), 0.0)
        self.assertEqual(
            float(ContinuousMassDiffusion(config).entropy_drift(mass, gamma, alpha)), 0.0
        )

    def test_packet_entropy_decreases_near_the_cutoff(self):
        model = PacketJumpDiffusion(PhysicsConfig())
        rate = model.entropy_drift(torch.tensor(0.26), torch.tensor(4.0))
        self.assertLess(float(rate), 0.0)

    def test_capped_moments_match_independent_survival_quadrature(self):
        # E[min(X,u)^r] = integral_0^u r*t^(r-1)*P(X>t) dt.
        marks, upper = PlanckNumberMarks(k_max=32), 0.4
        nodes, weights = np.polynomial.legendre.leggauss(96)
        times = (nodes + 1.0) * upper / 2.0
        components = np.arange(1, 33, dtype=float)
        mixture = components**-3 / np.sum(components**-3)
        z = times[:, None] * components
        survival = np.sum(mixture * np.exp(-z) * (1.0 + z + z**2 / 2.0), axis=1)
        for order in (1, 2):
            expected = upper / 2.0 * np.sum(weights * order * times ** (order - 1) * survival)
            self.assertAlmostEqual(
                float(marks.capped_moment(order, torch.tensor(upper))), float(expected), places=12
            )

    def test_numerical_epsilon_does_not_change_the_sampled_planck_law(self):
        low, high = PlanckNumberMarks(k_max=32, eps=1e-12), PlanckNumberMarks(k_max=32, eps=0.5)
        generator1, generator2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        for _ in range(12):
            torch.testing.assert_close(
                low.sample_dimensionless(torch.tensor(1.0), generator1),
                high.sample_dimensionless(torch.tensor(1.0), generator2),
                rtol=0,
                atol=0,
            )

    def test_engine_uses_zero_rate_for_absorbed_grid_intervals(self):
        config = PhysicsConfig(horizon=0.02, dt=0.01, planck_mixture_terms=32)
        theta = torch.tensor([1.0, 4.0, 0.1], requires_grad=True)
        path = JumpPath(
            torch.tensor([0.005]),
            torch.tensor([0.75]),
            torch.tensor([True]),
            torch.tensor(0.25),
            True,
            0.02,
        )
        engine = CentralizedInformationEngine(config)
        seen = []
        original = point_process_information

        def capture(full, reduced, dt, eps):
            seen.append(full.detach())
            return original(full, reduced, dt, eps)

        with (
            mock.patch.object(engine.jump, "simulate", return_value=path),
            mock.patch(
                "black_hole_information.engine.point_process_information", side_effect=capture
            ),
        ):
            engine.evaluate(theta, replicates=2, seed=4)
        torch.testing.assert_close(seen[0][:, 1], torch.zeros(2))


class MatchingTests(unittest.TestCase):
    def test_matched_paths_are_independent_of_requested_alpha(self):
        config = PhysicsConfig(horizon=0.02, dt=0.01, planck_mixture_terms=32)
        model = ContinuousMassDiffusion(config, moment_matched=True)
        first = torch.tensor([4.0, 4.0, 0.1])
        second = torch.tensor([4.0, 4.0, 1.0])
        path1 = model.simulate(first, torch.Generator().manual_seed(17))
        path2 = model.simulate(second, torch.Generator().manual_seed(17))
        torch.testing.assert_close(path1.masses, path2.masses, rtol=0, atol=0)

    def test_matched_drift_uses_the_packet_first_moment(self):
        config = PhysicsConfig(planck_mixture_terms=32)
        theta = torch.tensor([4.0, 4.0, 0.1])
        model = ContinuousMassDiffusion(config, moment_matched=True)
        observed = model.drift(theta[0], theta[2], theta[1])
        expected = (
            -theta[1] * mean_drift_matching(config, theta)["alpha_per_gamma"] / theta[0].square()
        )
        torch.testing.assert_close(observed, expected)


class LikelihoodTests(unittest.TestCase):
    def test_censored_girsanov_ratio_matches_independent_erfc(self):
        config = PhysicsConfig(horizon=0.01, dt=0.01, sigma_base=0.75)
        theta = torch.tensor([1.0, 4.0, 0.75], requires_grad=True)
        path = one_step_path(theta, config, hit=True)
        reported = ContinuousMassDiffusion(config).girsanov_loglik(path, theta, theta[0].detach())
        initial = -math.log(config.initial_observation_sd * math.sqrt(2 * math.pi))
        # z_ref=-10, z_target=-9.9; both probabilities are below the old epsilon.
        expected = math.log(math.erfc(9.9 / math.sqrt(2))) - math.log(
            math.erfc(10.0 / math.sqrt(2))
        )
        self.assertAlmostEqual(float(reported.detach()) - initial, expected, places=11)
        score = torch.autograd.grad(reported, theta)[0]
        self.assertGreater(float(score[2]), 0.0)

    def test_normalized_euler_score_retains_initial_mass_variance_term(self):
        config = PhysicsConfig(horizon=0.01, dt=0.01, planck_mixture_terms=32)
        theta = torch.tensor([4.0, 4.0, 0.4], requires_grad=True)
        path = one_step_path(theta, config)
        empty = torch.empty(0)
        packet = JumpPath(empty, empty, torch.empty(0, dtype=torch.bool), theta[0], False, 0.01)
        engine = CentralizedInformationEngine(config)
        with (
            mock.patch.object(engine.jump, "simulate", return_value=packet),
            mock.patch.object(engine.diffusion, "simulate", return_value=path),
            mock.patch("black_hole_information.engine.torch.randn", return_value=torch.tensor(0.0)),
        ):
            result = engine.evaluate(theta, replicates=2, seed=3)
        # At dm=mean and y0=M0, d log Normal(dm;mean, sigma/M0 sqrt(dt))/dM0=1/M0.
        self.assertAlmostEqual(float(result["fisher_b_native"][0, 0]), 1.0 / 4.0**2, places=9)

    def test_euler_likelihood_matches_normal_density_and_finite_difference(self):
        config = PhysicsConfig(horizon=0.01, dt=0.01)
        theta = torch.tensor([4.0, 4.0, 0.4], requires_grad=True)
        path = one_step_path(theta, config).detached()
        y0 = torch.tensor(4.01)
        model = ContinuousMassDiffusion(config)
        observed = model.euler_path_loglik(path, theta, y0)
        increment = path.masses[1] - path.masses[0]
        mean = -theta[2] / theta[0].square() * config.horizon
        sd = config.sigma_base / theta[0] * config.horizon**0.5
        expected = torch.distributions.Normal(theta[0], config.initial_observation_sd).log_prob(y0)
        expected = expected + torch.distributions.Normal(mean, sd).log_prob(increment)
        torch.testing.assert_close(observed, expected)
        score = torch.autograd.grad(observed, theta)[0]
        for index in (0, 2):
            plus, minus = theta.detach().clone(), theta.detach().clone()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            finite_difference = (
                model.euler_path_loglik(path, plus, y0) - model.euler_path_loglik(path, minus, y0)
            ) / 2e-5
            torch.testing.assert_close(score[index], finite_difference, rtol=1e-5, atol=1e-7)

    def test_euler_hitting_atom_and_impossible_regular_support(self):
        config = PhysicsConfig(horizon=0.01, dt=0.01, sigma_base=0.75)
        theta = torch.tensor([1.0, 4.0, 0.75], requires_grad=True)
        model = ContinuousMassDiffusion(config)
        observed = model.euler_path_loglik(
            one_step_path(theta, config, hit=True), theta, theta[0].detach()
        )
        expected = -math.log(config.initial_observation_sd * math.sqrt(2 * math.pi)) + math.log(
            math.erfc(9.9 / math.sqrt(2)) / 2
        )
        self.assertAlmostEqual(float(observed.detach()), expected, places=11)
        regular = DiffusionPath(
            torch.tensor([0.0, 0.01]),
            torch.tensor([1.0, 0.9]),
            torch.tensor([True]),
            torch.tensor([False]),
        )
        candidate = torch.tensor([0.3, 4.0, 0.75], requires_grad=True)
        self.assertTrue(
            torch.isneginf(model.euler_path_loglik(regular, candidate, torch.tensor(1.0)))
        )

    def test_malformed_path_masks_are_rejected(self):
        config = PhysicsConfig(horizon=0.01, dt=0.01)
        theta = torch.tensor([1.0, 4.0, 0.1], requires_grad=True)
        path = one_step_path(theta, config)
        path.hit_transition = torch.tensor([True])  # Cannot be both regular and censored.
        with self.assertRaisesRegex(ValueError, "transition"):
            ContinuousMassDiffusion(config).girsanov_loglik(path, theta, theta[0].detach())

    def test_jump_likelihood_rejects_out_of_order_events(self):
        config = PhysicsConfig(horizon=0.1, dt=0.01)
        theta = torch.tensor([1.0, 4.0, 0.1], requires_grad=True)
        path = JumpPath(
            torch.tensor([0.08, 0.02]),
            torch.tensor([0.1, 0.1]),
            torch.tensor([False, False]),
            torch.tensor(0.8),
            False,
            0.1,
        )
        with self.assertRaisesRegex(ValueError, "event times"):
            PacketJumpDiffusion(config).complete_path_loglik(path, theta, theta[0].detach())


class InformationSupportTests(unittest.TestCase):
    def test_dominant_replicate_does_not_cancel_other_positive_predictors(self):
        values = torch.tensor([[1e30], [1.0], [1.0]])
        self.assertEqual(float(_leave_one_out_mean(values)[0]), 1.0)

    def test_positive_full_with_zero_reduced_has_infinite_information(self):
        value = point_process_information(torch.ones(1), torch.zeros(1), dt=0.1)
        self.assertTrue(torch.isposinf(value))

    def test_zero_full_rates_have_exact_limits_and_finite_model_gradients(self):
        theta = torch.tensor(1.0, requires_grad=True)
        full = theta * torch.zeros(2)
        self.assertEqual(
            float(point_process_information(full, torch.tensor([0.0, 2.0]), dt=0.1).detach()), 0.2
        )
        value = point_process_information(full, torch.zeros(2), dt=0.1)
        self.assertEqual(float(value.detach()), 0.0)
        self.assertEqual(float(torch.autograd.grad(value, theta)[0]), 0.0)

    def test_engine_rejects_unsupported_predictors_before_publishing_gradient(self):
        config = PhysicsConfig(horizon=0.01, dt=0.01)
        theta = torch.tensor([4.0, 4.0, 0.4], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "zero.*positive|support"):
            CentralizedInformationEngine(config).evaluate(
                theta, replicates=2, reduced_rate_a=torch.zeros((2, 1))
            )

    def test_positive_transform_rejects_unrepresentable_parameters(self):
        with self.assertRaisesRegex(ValueError, "representable|physical"):
            positive_parameters(torch.full((3,), -1000.0), 0.25)


if __name__ == "__main__":
    unittest.main()
