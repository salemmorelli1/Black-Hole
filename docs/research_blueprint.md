# A Statistical Information Theory of Evaporating Black Holes

## Research blueprint for a PhD thesis in Statistics

### Scope and status

This thesis treats semiclassical black-hole evaporation as a stopped, nonstationary
stochastic process rather than an equilibrium time series.  It compares a marked
jump model (Paradigm A) with an Itô diffusion (Paradigm B), estimates trajectory-level
Fisher information, and quantifies directed information from latent mass to a detector
filtration.  All equations use Planck units,

\[
G=c=\hbar=k_B=1,
\qquad T_H(M)=\frac{\kappa_H}{M},
\qquad \kappa_H=\frac{1}{8\pi},
\qquad S_{\mathrm{BH}}(M)=4\pi M^2.
\]

Neither effective model is asserted to remain valid at \(M=0\).  Define a physically
interpretable cutoff \(m_\star>0\), the stopping time

\[
\tau_\star^P=\inf\{t\ge 0:M_t^P\le m_\star\},\qquad P\in\{A,B\},
\]

and analyze the stopped path \(M_{t\wedge\tau_\star^P}^P\) on a fixed interval
\([0,T]\).  The endpoint is censored or absorbed; it is not called a resolved
singularity.

---

## 1. Thesis estimand and the necessary control design

The proposed physical hypothesis is

> A smooth continuum representation suppresses information-theoretic structure that
> remains visible when evaporation is represented as discrete stochastic emissions.

Let \(I_P(T)\) denote integrated continuous-time transfer entropy from latent mass to
the detector under paradigm \(P\):

\[
I_P(T)=\int_0^{T\wedge\tau_\star^P}\dot{\mathcal T}_P(t)\,dt,
\qquad
\Delta_I(T)=I_B(T)-I_A(T).
\]

The directional thesis alternative is

\[
H_1:\;\mathbb E\{\Delta_I(T)\}<0,
\]

with \(H_0:\mathbb E\{\Delta_I(T)\}=0\).  A negative residual means that the
continuous model transmits less information to the observation filtration.

### 1.1 Two confounds in the originally specified comparison

The raw A-versus-B comparison does **not**, by itself, isolate coarse-graining.

1. **The infinitesimal variances do not match.** Under the stated Planck mark law,
   Paradigm A has quadratic-variation rate proportional to \(M^{-3}\); the requested
   Paradigm B coefficient \(\sigma_{\rm base}/M\) has rate proportional to \(M^{-2}\).
2. **The detector laws differ.** An event-time/mark detector and a continuous Gaussian
   telemetry channel do not generate the same statistical experiment.  Their
   information difference contains both latent-dynamics and observation-channel
   effects.

The thesis should therefore report three comparisons:

- **Primary phenomenological comparison:** the requested A model versus the requested
  B model.
- **Moment-matched control:** A versus \(B^\star\), whose drift and infinitesimal
  variance match the first two local moments of A.
- **Common-detector control:** pass both latent models through the same calibrated
  detector operator.  A 2-by-2 design (latent law: jump/diffusion; detector:
  event/continuous) separates latent granularity from detector coarse-graining.

Only the moment-matched, common-detector contrast supports the narrow causal statement
that a detected information gap is attributable to smoothing rather than to a changed
variance law or measurement system.

---

## 2. Unified measure-theoretic framework

Let

\[
(\Omega,\mathcal F,\mathbb F,\mathbb P_\vartheta),
\qquad
\mathbb F=(\mathcal F_t)_{0\le t\le T},
\qquad
\vartheta=(M_0,\gamma,\alpha)^\top,
\]

be a complete, right-continuous filtered probability space satisfying the usual
conditions.  The canonical space supports:

- a marked integer-valued random measure \(N(dt,de)\);
- independent Brownian motions \(W\) and \(V\);
- parameter-free base uniforms and Gaussian innovations used for reparameterized
  simulation;
- an initial noisy calibration measurement \(Y_0=M_0+\epsilon_0\), with
  \(\epsilon_0\sim N(0,r_0^2)\).

The initial observation is statistically important.  If \(M_0\) is a deterministic
initial condition observed without noise, path laws indexed by different \(M_0\) may
be non-dominated, so an ordinary Fisher information for \(M_0\) need not exist.

For each paradigm define:

\[
\mathcal X_t^P=\sigma(M_s^P:0\le s\le t),
\qquad
\mathcal Y_t^P=\sigma(Y_s^P:0\le s\le t),
\qquad
\mathcal F_t^P=\mathcal X_t^P\vee\mathcal Y_t^P\vee\mathcal N,
\]

where \(\mathcal N\) contains the \(\mathbb P_\vartheta\)-null sets.  Transfer
entropy conditions on \(\mathcal Y_{t-}^P\), whereas complete-data likelihoods
condition on the larger filtration containing the latent path.  Those are distinct
statistical objects and must remain distinct in notation and software.

---

## 3. Paradigm A: discrete packet jump process

### 3.1 State equation and compensator

Let the predictable marked intensity be

\[
\nu_A(dt,de\mid\mathcal F_{t-})
=\mathbf 1_{\{M_{t-}^A>m_\star\}}
\lambda_\gamma(M_{t-}^A)f_E(e\mid M_{t-}^A)\,de\,dt,
\qquad
\lambda_\gamma(M)=\frac{\gamma}{M}.
\]

The stopped mass process is

\[
M_t^A=M_0-
\int_{(0,t\wedge\tau_\star^A]\times(0,\infty)}
\min\{e,M_{s-}^A-m_\star\}\,N(ds,de).
\]

A jump that would cross the cutoff is represented by a terminal boundary atom.  This
avoids negative mass while retaining the tail probability of the mark distribution.

### 3.2 Thermal mark law

For a baseline massless-boson number spectrum, write \(E=T_H(M)X\), where

\[
f_X(x)=\frac{x^2}{2\zeta(3)(e^x-1)},\qquad x>0.
\]

The dimensionless moments are

\[
c_r=\mathbb E(X^r)
=\frac{\Gamma(r+3)\zeta(r+3)}{2\zeta(3)},
\]

so

\[
c_1=\frac{\pi^4}{30\zeta(3)}\approx2.701178,
\qquad
c_2=\frac{12\zeta(5)}{\zeta(3)}\approx10.35153.
\]

A greybody-corrected model replaces the numerator by
\(G(e,M)e^2\), normalizes the density, and estimates or fixes the parameters of
\(G\).  The uncorrected Planck spectrum is a baseline model, not a final statement
about the exact Hawking emission spectrum.

The PyTorch implementation uses the exact mixture identity

\[
K\sim p_k\propto k^{-3},
\qquad
X\mid K=k\sim\operatorname{Gamma}(3,k),
\]

truncated at a configurable large \(k_{\max}\).  Because \(K\) is parameter-free and
\(E=T_H(M)X\), gradients with respect to physical parameters flow through the thermal
scale without differentiating a parameter-dependent categorical distribution.

### 3.3 Generator, mean flux, and variance

For a regular test function \(g\), away from the terminal boundary,

\[
(\mathcal L_A g)(M)
=\frac{\gamma}{M}\int_0^\infty
\{g(M-e)-g(M)\}f_E(e\mid M)\,de.
\]

The first two local moments are

\[
\frac{\mathbb E(dM_t^A\mid\mathcal F_{t-})}{dt}
=-\frac{\gamma\kappa_Hc_1}{M_{t-}^2},
\]

\[
\frac{\mathbb E\{(dM_t^A)^2\mid\mathcal F_{t-}\}}{dt}
=\frac{\gamma\kappa_H^2c_2}{M_{t-}^3}.
\]

Therefore

\[
\alpha_{\rm eff}=\gamma\kappa_Hc_1
\]

matches the Hawking drift.  A globally moment-matched diffusion must use

\[
\sigma_\star(M)
=\sqrt{\gamma\kappa_H^2c_2}\,M^{-3/2},
\]

not \(\sigma_{\rm base}M^{-1}\).

### 3.4 Entropy dissipation

The predictable Bekenstein-Hawking entropy drift is

\[
(\mathcal L_A S_{\rm BH})(M)
=\frac{\gamma}{M}\,
\mathbb E\left[4\pi\{(M-E)^2-M^2\}\mid M\right],
\]

and, away from the cutoff,

\[
(\mathcal L_A S_{\rm BH})(M)
=\frac{\gamma}{M}
\left(-8\pi M\frac{\kappa_Hc_1}{M}
+4\pi\frac{\kappa_H^2c_2}{M^2}\right).
\]

This thermodynamic entropy drift is not automatically equal to recoverable quantum
information.  It is a physical state functional to be analyzed jointly with, but not
substituted for, transfer entropy.

### 3.5 Time-rescaling simulation and gradient qualification

Between jumps, \(M\) is constant, so

\[
\Delta t_i=\frac{-\log U_i}{\lambda_\gamma(M_{t_{i-1}})},
\qquad U_i\sim\operatorname{Uniform}(0,1),
\]

and

\[
E_i=T_H(M_{t_i-})X_i.
\]

This produces an almost-everywhere pathwise derivative conditional on the realized
number and order of events.  It does not differentiate the probability that a path
changes event count or crosses the terminal boundary.  Consequently:

- use pathwise autograd for smooth conditional sensitivities and variance reduction;
- use fixed-data likelihood scores, score-function corrections, or weak derivatives
  for Fisher information and unbiased gradients involving event topology;
- never claim that inverse time rescaling alone solves all discontinuous-gradient
  problems.

### 3.6 Complete-path likelihood

For observed latent events \((t_i,E_i)_{i=1}^{n_T}\), before a terminal atom,

\[
\ell_A(\vartheta)
=\sum_{i=1}^{n_T}\left[
\log\lambda_\gamma(M_{t_i-})
+\log f_E(E_i\mid M_{t_i-})
\right]
-\int_0^{T\wedge\tau_\star^A}\lambda_\gamma(M_s)\,ds
+\ell_0(M_0).
\]

For a terminal event, replace the continuous mark log density by
\(\log\Pr\{E\ge M_{t_i-}-m_\star\mid M_{t_i-}\}\).  The compensator is exactly
piecewise constant between events.

---

## 4. Paradigm B: continuous quantum-field flow

### 4.1 State equation

The requested stopped Itô model is

\[
dM_t^B=-\frac{\alpha}{(M_t^B)^2}\,dt
+\frac{\sigma_{\rm base}}{M_t^B}\,dW_t,
\qquad t<\tau_\star^B.
\]

The model is defined only on \([m_\star,\infty)\).  The coefficients are locally
Lipschitz there, so the stopped SDE avoids the unresolved singular endpoint.

Euler-Maruyama gives

\[
M_{k+1}=M_k-\frac{\alpha}{M_k^2}\Delta t
+\frac{\sigma_{\rm base}}{M_k}\sqrt{\Delta t}\,\xi_k,
\qquad \xi_k\stackrel{\mathrm{iid}}{\sim}N(0,1),
\]

followed by absorption/censoring at \(m_\star\).  A stable grid must obey the local
scales

\[
\Delta t\ll\frac{M^3}{\alpha},
\qquad
\Delta t\ll\frac{M^4}{\sigma_{\rm base}^2}.
\]

Thus a fixed coarse time step becomes invalid near the cutoff even when every tensor
remains finite.

The control model \(B^\star\) retains the same drift but replaces the diffusion by

\[
dM_t^{B^\star}=-\frac{\alpha_{\rm eff}}{(M_t^{B^\star})^2}\,dt
+\frac{\sqrt{\gamma\kappa_H^2c_2}}{(M_t^{B^\star})^{3/2}}\,dW_t.
\]

### 4.2 Girsanov path likelihood

Let \(b_\alpha(M)=-\alpha/M^2\), \(s(M)=\sigma_{\rm base}/M\), and
\(a(M)=s^2(M)\).  Relative to a zero-drift reference diffusion with the same
\(s(M)\), Girsanov's theorem gives

\[
\ell_B(\vartheta)-\ell_{B,0}
=\int_0^{T\wedge\tau_\star^B}\frac{b_\alpha(M_t)}{a(M_t)}\,dM_t
-\frac12\int_0^{T\wedge\tau_\star^B}
\frac{b_\alpha^2(M_t)}{a(M_t)}\,dt.
\]

The Euler approximation used in software is

\[
\ell_B-\ell_{B,0}
\approx\sum_k\left[
\frac{b_k}{a_k}\Delta M_k-
\frac12\frac{b_k^2}{a_k}\Delta t
\right].
\]

When an Euler increment reaches \(m_\star\), the implementation uses the ratio of
Gaussian boundary-hitting probabilities instead of treating the censored value as an
ordinary Gaussian observation.  Girsanov requires the same diffusion coefficient
under target and reference measures.  If \(\sigma\) is itself parameterized, the
path-measure problem changes and this likelihood cannot be reused unchanged.

### 4.3 Entropy drift

Itô's formula gives

\[
dS_{\rm BH}(M_t)
=8\pi M_t\,dM_t+4\pi(dM_t)^2.
\]

For the requested B model,

\[
\mathbb E\{dS_{\rm BH}(M_t)\mid\mathcal F_t\}/dt
=-\frac{8\pi\alpha}{M_t}
+\frac{4\pi\sigma_{\rm base}^2}{M_t^2}.
\]

The second term is an Itô correction.  Omitting it would systematically misstate the
expected entropy drift.

---

## 5. Observation filtrations and transfer entropy

### 5.1 Marked event telemetry

Let the detector observe a marked point measure \(D(dt,dz)\) with full predictable
intensity density \(\nu_t^\star(z)\), which may include thinning, background events,
and mark noise.  The reduced prediction based only on detector history is

\[
\bar\nu_t(z)=
\mathbb E\{\nu_t^\star(z)\mid\mathcal Y_{t-}^A\}.
\]

The marked point-process transfer-entropy rate is

\[
\dot{\mathcal T}_A(t)
=\mathbb E\left[
\int\left{
\nu_t^\star(z)\log\frac{\nu_t^\star(z)}{\bar\nu_t(z)}
-\nu_t^\star(z)+\bar\nu_t(z)
\right}dz
\right].
\]

For count-only telemetry, the mark integral reduces to

\[
\dot{\mathcal T}_A(t)
=\mathbb E\left[
\lambda_t^\star\log\frac{\lambda_t^\star}{\bar\lambda_t}
-\lambda_t^\star+\bar\lambda_t
\right].
\]

### 5.2 Continuous Gaussian telemetry

Use

\[
dY_t^B=h(M_t^B)\,dt+\rho\,dV_t,
\qquad
h(M)=\frac{g_{\rm obs}}{M^2},
\]

and define the observation-filtration predictor

\[
\widehat h_t=\mathbb E\{h(M_t^B)\mid\mathcal Y_t^B\}.
\]

The Gaussian-channel transfer-entropy rate is

\[
\dot{\mathcal T}_B(t)
=\frac{1}{2\rho^2}
\mathbb E\left[\{h(M_t^B)-\widehat h_t\}^2\right].
\]

This is the continuous-time likelihood-ratio identity underlying the filtering
interpretation of mutual/directed information.

### 5.3 Naming rule enforced in software

The equations above are transfer entropy only if \(\bar\nu_t\) and \(\widehat h_t\)
are causal, observation-history predictions.  Replacing them by an unconditional
ensemble mean gives a legitimate open-loop likelihood-information contrast, but not
transfer entropy.  The reference implementation therefore:

- accepts user-supplied predictable filter outputs;
- labels the default leave-one-out ensemble baseline as **not transfer entropy**;
- prevents a convenient numerical surrogate from being reported under the stronger
  theoretical name.

The final thesis implementation should estimate the reduced predictors with a causal
particle/Snyder filter for A and a particle or assumed-density nonlinear filter for B.
Filter approximation error must be propagated into uncertainty for \(\Delta_I\).

### 5.4 Common-detector experiment

To isolate the latent-law effect, define a common detector response kernel \(K\).  For
A, convolve emitted packet energy with \(K\); for B, convolve the continuous emitted
power with the same \(K\).  Add the same bandwidth, detection efficiency, background,
and electronics noise.  Compare both models under:

1. event-resolving telemetry;
2. temporally aggregated continuous telemetry.

The interaction between latent law and detector law is the statistically defensible
measure of information masked specifically by macroscopic observation.

---

## 6. Fisher information and the information residual

### 6.1 Definition

For a dominated path experiment with log likelihood \(\ell_P(\vartheta;\mathcal D)\),

\[
s_P(\vartheta)=\nabla_\vartheta\ell_P(\vartheta;\mathcal D),
\qquad
\mathcal J_P(\vartheta)=\mathbb E_\vartheta
\{s_P(\vartheta)s_P(\vartheta)^\top\}.
\]

The Monte Carlo estimator is

\[
\widehat{\mathcal J}_P
=\frac1R\sum_{r=1}^R s_{P,r}s_{P,r}^\top.
\]

When simulating a path as \(\mathcal D=g_\vartheta(U)\), the score must be computed as
the partial derivative of \(\ell(\vartheta;\mathcal D)\) with \(\mathcal D\) fixed.
Differentiating through \(g_\vartheta\) produces a pathwise sensitivity, not a score.
The software detaches each realized path before evaluating its Fisher score.

### 6.2 Analytic complete-data information rates

For a marked point process,

\[
\mathcal J_A
=\mathcal J_0+
\mathbb E\int_0^{T\wedge\tau_\star^A}\int
\nabla_\vartheta\log\nu_A(t,e)
\nabla_\vartheta\log\nu_A(t,e)^\top
\nu_A(t,e)\,de\,dt,
\]

with an additional discrete contribution for a terminal boundary atom.  The direct
arrival-rate contribution for \(\gamma\) is

\[
\mathcal J_{A,\gamma\gamma}^{(\lambda)}
=\mathbb E\int_0^{T\wedge\tau_\star^A}\frac{dt}{\gamma M_t^A}.
\]

For B with known parameter-independent diffusion coefficient,

\[
\mathcal J_B
=\mathcal J_0+
\mathbb E\int_0^{T\wedge\tau_\star^B}
\frac{\nabla_\vartheta b_\alpha(M_t)
\nabla_\vartheta b_\alpha(M_t)^\top}{a(M_t)}\,dt.
\]

In particular,

\[
\mathcal J_{B,\alpha\alpha}
=\mathbb E\int_0^{T\wedge\tau_\star^B}
\frac{dt}{\sigma_{\rm base}^2M_t^2}.
\]

These are complete-path quantities.  Observed-data information is smaller unless the
detector recovers the latent path exactly; it requires the marginal observation
likelihood or a valid Fisher/Louis identity calculation under the state-space model.

### 6.3 Structural singularity of the native parameter vector

With \(\vartheta=(M_0,\gamma,\alpha)\):

- \(\alpha\) is inactive in A;
- \(\gamma\) is inactive in the requested B.

Therefore both native 3-by-3 Fisher matrices are structurally singular, and the raw
matrix difference \(\mathcal J_B-\mathcal J_A\) is not a meaningful measure of lost
information about the same parameter.

Define the common parameter

\[
\psi=(M_0,\alpha_{\rm eff})^\top,
\qquad
\gamma=\frac{\alpha_{\rm eff}}{\kappa_Hc_1}
\quad\text{in A},
\qquad
\alpha=\alpha_{\rm eff}
\quad\text{in B}.
\]

With Jacobians \(G_A=\partial\vartheta_A/\partial\psi\) and
\(G_B=\partial\vartheta_B/\partial\psi\), compare the pullbacks

\[
\mathcal J_A^{(\psi)}=G_A^\top\mathcal J_AG_A,
\qquad
\mathcal J_B^{(\psi)}=G_B^\top\mathcal J_BG_B,
\]

and define

\[
\Delta_{\mathcal J}=\mathcal J_B^{(\psi)}-\mathcal J_A^{(\psi)}.
\]

The difference need not be positive or negative semidefinite.  Report its eigenvalues,
generalized eigenvalues, trace, and regularized log-determinant contrast rather than
reducing it automatically to a single unsigned norm.

---

## 7. Proof sketches for the central claims

### Proposition 1: mean-drift calibration

Because \(E=T_H(M)X=\kappa_HX/M\),

\[
\mathbb E(E\mid M)=\frac{\kappa_Hc_1}{M}.
\]

Multiplying by \(\lambda(M)=\gamma/M\) gives

\[
\mathbb E(dM\mid M)/dt
=-\lambda(M)\mathbb E(E\mid M)
=-\gamma\kappa_Hc_1M^{-2}.
\]

Hence \(\alpha_{\rm eff}=\gamma\kappa_Hc_1\).  Similarly,

\[
\lambda(M)\mathbb E(E^2\mid M)
=\gamma\kappa_H^2c_2M^{-3},
\]

which proves that the globally matched diffusion coefficient scales as \(M^{-3/2}\).

### Proposition 2: Girsanov score

For two stopped diffusions with common nonzero diffusion \(s(M)\) and drifts \(b\)
and \(b_0\), the exponential martingale condition on the stopped interval yields

\[
\log\frac{d\mathbb P_b}{d\mathbb P_{b_0}}
=\int\frac{b-b_0}{a}\,dM
-\frac12\int\frac{b^2-b_0^2}{a}\,dt.
\]

Setting \(b_0=0\) gives the likelihood used above.  Stopping at \(m_\star\) keeps the
coefficients bounded on each compact mass interval and makes the change-of-measure
conditions materially more defensible than an unstopped calculation at \(M=0\).

### Proposition 3: continuous-time transfer-entropy integrands

For a counting observation, the log Radon-Nikodym derivative between full and reduced
predictable intensities is

\[
\int\log(\lambda^\star/\bar\lambda)\,dN
-\int(\lambda^\star-\bar\lambda)\,dt.
\]

Taking expectation and using
\(\mathbb E(dN\mid\mathcal F_{t-})=\lambda^\star dt\) produces
\(\lambda^\star\log(\lambda^\star/\bar\lambda)-\lambda^\star+\bar\lambda\).
The marked formula follows by integrating over the mark coordinate.

For \(dY=h\,dt+\rho\,dV\), the corresponding Gaussian likelihood ratio has expected
quadratic term \(\tfrac12(h-\widehat h)^2/\rho^2\).  Conditioning the reduced drift on
the observation filtration produces the stated continuous-time directed-information
rate.

### Proposition 4: Fisher pullback

Under a differentiable reparameterization \(\vartheta=g(\psi)\), the chain rule gives

\[
s_\psi=(\partial g/\partial\psi)^\top s_\vartheta.
\]

Taking the expected outer product gives

\[
\mathcal J_\psi=G^\top\mathcal J_\vartheta G.
\]

Thus Fisher matrices from A and B must be pulled back to a shared parameterization
before subtraction.

---

## 8. PyTorch architecture

The accompanying module implements:

- a finite-mixture Planck sampler and internally consistent mark likelihood;
- event-time simulation by inverse integrated hazard;
- a terminal mark atom at \(m_\star\);
- exact piecewise-constant jump compensators;
- Euler-Maruyama simulation for the requested B and moment-matched \(B^\star\);
- a discrete Girsanov likelihood with a censored boundary-hitting term;
- Bekenstein-Hawking entropy drifts for both paradigms;
- fixed-data autograd scores and empirical Fisher matrices;
- common-parameter Fisher pullbacks;
- point-process and Gaussian-channel information functionals;
- explicit labels distinguishing transfer entropy from open-loop surrogates;
- pathwise gradients reported separately from likelihood scores.

Minimal execution:

```python
import torch

from black_hole_information_engine import (
    CentralizedInformationEngine,
    PhysicsConfig,
    mean_drift_matching,
)

torch.set_default_dtype(torch.float64)

config = PhysicsConfig(
    horizon=1.0,
    dt=1.0e-3,
    m_floor=0.25,
    sigma_base=0.02,
    initial_observation_sd=0.05,
    observation_sd=0.10,
)

# Physical scale: theta = (M0, gamma, alpha).
theta = torch.tensor([4.0, 2.0, 0.20], requires_grad=True)

engine = CentralizedInformationEngine(config)
result = engine.evaluate(theta, replicates=512, seed=20260820)

print(mean_drift_matching(config, theta))
print(result["fisher_a_native"])
print(result["fisher_b_native"])
print(result["delta_fisher_common_B_minus_A"])
print(result["delta_information_B_minus_A_nats"])
print(result["delta_information_pathwise_gradient"])
print(result["a_predictor_status"], result["b_predictor_status"])
```

For a thesis result labeled transfer entropy, pass filter-based tensors
``reduced_rate_a`` and ``reduced_signal_b`` to ``evaluate``.  The default ensemble
baselines are intentionally labeled as open-loop contrasts.

### 8.1 HMC/NUTS interface

Use an unconstrained parameter vector \(\eta\in\mathbb R^3\) and

\[
M_0=m_\star+\operatorname{softplus}(\eta_0),\quad
\gamma=\operatorname{softplus}(\eta_1),\quad
\alpha=\operatorname{softplus}(\eta_2).
\]

The module returns the log-Jacobian

\[
\sum_{j=0}^2\log\operatorname{sigmoid}(\eta_j)
\]

for use in an unconstrained log posterior.  A preferable scientific
parameterization is

\[
(\log M_0,\log\gamma,\delta),
\qquad
\delta=\log\frac{\alpha}{\gamma\kappa_Hc_1},
\]

where \(\delta=0\) represents drift matching and \(\delta\) estimates model
discrepancy.

---

## 9. Critical identifiability and HMC/NUTS analysis

### 9.1 Structural non-identifiability

The native parameter vector is overcomplete within each paradigm.  No amount of data
from A identifies an unused \(\alpha\), and no amount from the requested B identifies
an unused \(\gamma\).  A joint posterior must either:

- use paradigm-specific parameter vectors;
- impose the physical bridge \(\alpha=\gamma\kappa_Hc_1\); or
- estimate the discrepancy parameter \(\delta\) above.

Otherwise the posterior contains exact flat directions and NUTS cannot repair the
model by tuning.

### 9.2 Lifetime confounding

Ignoring diffusion, the Hawking drift has solution

\[
M(t)=\{M_0^3-3\alpha t\}^{1/3},
\qquad
\tau_0=\frac{M_0^3}{3\alpha}.
\]

Sparse observations of only the lifetime identify approximately the ratio
\(M_0^3/\alpha\), producing a strong posterior ridge.  Early direct calibration of
\(M_0\), multiple intermediate observations, and informative scale priors are needed
to separate \(M_0\) from \(\alpha\).

In A, count-only telemetry primarily identifies \(\gamma/M\).  Energy marks supply
temperature information proportional to \(1/M\) and are therefore essential for
separating \(\gamma\) from mass.  Aggregating away the marks creates precisely the
identifiability loss the thesis seeks to measure.

### 9.3 The apparent information explosion is not automatically infinite

Although local rates diverge as \(M\downarrow0\), the remaining physical time shrinks.
Under the deterministic drift approximation,

\[
dt=-\frac{M^2}{\alpha}\,dM.
\]

Then

\[
\mathcal J_{B,\alpha\alpha}
\approx\int_{m_\star}^{M_0}
\frac{1}{\sigma_{\rm base}^2M^2}\frac{M^2}{\alpha}\,dM
=\frac{M_0-m_\star}{\alpha\sigma_{\rm base}^2},
\]

which is finite.  Likewise, the direct \(\gamma\)-information in A is approximately

\[
\int\frac{dt}{\gamma M}
\approx\frac{M_0^2-m_\star^2}{2\gamma\alpha_{\rm eff}},
\]

also finite.  The final regime can be numerically explosive without containing
infinite integrated statistical information.

### 9.4 HMC geometry near the cutoff

The derivatives

\[
\frac{\partial b}{\partial M}=\frac{2\alpha}{M^3},
\qquad
\frac{\partial\sigma}{\partial M}=-\frac{\sigma_{\rm base}}{M^2}
\]

generate extreme local curvature.  Expected consequences are divergent transitions,
small adapted step sizes, maximum tree-depth hits, poor energy behavior, and strongly
anisotropic posterior geometry.  Recommended controls are:

- stop at a scientifically justified \(m_\star\) and model the terminal observation
  as censoring;
- nondimensionalize mass by \(M_0\) and time by \(M_0^3/\alpha_{\rm ref}\);
- use log/softplus parameterizations and noncentered Brownian innovations;
- avoid a parameter-dependent adaptive mesh inside an HMC target unless its Jacobian
  and topology are handled explicitly;
- compare step-size refinements and require posterior stability as \(\Delta t\to0\);
- monitor divergences, tree depth, rank-normalized \(\widehat R\), effective sample
  size, and energy diagnostics;
- perform simulation-based calibration before interpreting posterior intervals.

### 9.5 Variable-dimension latent events

Vanilla HMC/NUTS operates on a fixed-dimensional continuous state.  If event times and
marks are observed, A has a fixed data dimension and its parameter likelihood is
smooth away from the boundary.  If emissions are latent, event count is unknown and
the posterior is trans-dimensional.  Padding to ``max_events`` does not remove the
topological discontinuity.

Valid approaches include marginalizing latent events, particle marginal methods,
reversible-jump updates, or a hybrid sampler that uses HMC only for the continuous
parameter block.  A pure NUTS claim for an unknown event count would be methodologically
incorrect.

### 9.6 Diffusion-state dimension

For B, sampling every latent state directly can create a high-dimensional funnel as
\(\Delta t\) decreases.  Use Brownian innovations
\(\xi_k\sim N(0,1)\) as the noncentered latent variables, or marginalize/filter the
state when an accurate approximation is available.  If \(\sigma_{\rm base}\) is
estimated from noisy discrete telemetry, separate process noise from detector noise;
otherwise the two variance components can be weakly identified.

---

## 10. Statistical study design

### 10.1 Primary estimands

Pre-register:

1. integrated transfer-entropy residual \(\Delta_I(T)\);
2. time-resolved residual \(\Delta_{\dot I}(t)\);
3. common-parameter Fisher residual \(\Delta_{\mathcal J}\);
4. entropy-drift difference
   \(\Delta_{\dot S}(t)=\dot S_B(t)-\dot S_A(t)\);
5. detector-latent interaction from the 2-by-2 control design.

### 10.2 Monte Carlo design

Use paired common random numbers for A and B where mathematically meaningful, but do
not force identical noise to imply identical laws.  For each parameter setting:

- generate independent outer replicates;
- fit the causal filters on independent or cross-fitted simulation sets;
- evaluate transfer entropy on held-out paths;
- report Monte Carlo standard errors and simultaneous confidence bands;
- repeat across \(m_\star\), \(\Delta t\), detector bandwidth, efficiency, and noise;
- include the moment-matched \(B^\star\) control.

For the scalar endpoint, use the paired replicate differences

\[
D_r=I_{B,r}(T)-I_{A,r}(T)
\]

and estimate \(\mathbb E(D_r)\) with a paired confidence interval.  For the functional
rate curve, use a simultaneous multiplier/bootstrap band rather than pointwise bands.

### 10.3 Falsification logic

Evidence for the thesis hypothesis requires all of the following:

- a materially negative \(\Delta_I\) under a common detector;
- persistence under moment matching and time-step refinement;
- filter-calibration diagnostics showing that the result is not a reduced-predictor
  artifact;
- uncertainty intervals excluding a pre-specified practically negligible region;
- robustness across plausible greybody corrections and cutoffs.

If the effect disappears after moment matching or common-detector calibration, the
original difference was produced by model or measurement mismatch rather than by
continuum smoothing.  That is a scientifically valuable falsification, not a failed
thesis.

---

## 11. Implementation milestones and acceptance criteria

### Phase I: mathematical validation

- Prove existence/uniqueness for each stopped model.
- Verify compensators and terminal likelihood terms.
- Prove the mean/variance matching equations.
- Specify the detector kernels and reduced filtrations.

### Phase II: computational validation

- Unit-test Planck moments against analytic constants.
- Verify score gradients by central finite differences with data held fixed.
- Verify pathwise gradients under fixed event topology.
- Test Fisher positive semidefiniteness and structural zero directions.
- Demonstrate weak convergence under \(\Delta t\) refinement.
- Measure terminal-event and ``max_events`` frequencies.

### Phase III: filtering and transfer entropy

- Implement a causal marked-point-process filter for A.
- Implement a nonlinear diffusion filter for B.
- Cross-fit filters and quantify approximation bias.
- Replace all open-loop placeholders before labeling results transfer entropy.

### Phase IV: inferential study

- Run the phenomenological, moment-matched, and common-detector comparisons.
- Report \(\Delta_I\), \(\Delta_{\mathcal J}\), uncertainty, and sensitivity analyses.
- Conduct simulation-based calibration for the posterior workflow.
- Release seeds, configurations, code, and environment lock files.

---

## 12. Reference basis

The blueprint is grounded in the supplied library, especially:

- Krishnan, *Quantum Field Theory, Black Holes and Holography*;
- Harlow, *Jerusalem Lectures on Black Holes and Quantum Information*;
- Tong, Carroll, Blau, Zhou, and Reall on general relativity;
- Dowker, Townsend, Reall, Strominger, and Dabholkar-Nampuri on black holes and
  quantum black holes;
- Gourgoulhon on the 3+1 formalism;
- Minguzzi on Lorentzian causality;
- Dafermos-Rodnianski on black holes and linear waves;
- Keener, *Theoretical Statistics: Topics for a Core Course*;
- Lehmann-Casella, *Theory of Point Estimation*;
- Lehmann-Romano, *Testing Statistical Hypotheses*;
- Shumway-Stoffer, *Time Series Analysis and Its Applications*, fifth edition;
- Robert, *The Bayesian Choice*, and Robert-Casella, *Monte Carlo Statistical
  Methods*;
- Karr, Chow-Teicher, Gut, and Shorack on probability;
- Cowles and Kostas on Bayesian computation and state-space models;
- Chan on statistical modeling and computation.

Foundational external results to cite in a thesis manuscript include Bekenstein on
black-hole entropy, Hawking on particle creation, Girsanov on changes of measure for
diffusions, Duncan on continuous-time Gaussian-channel information, Schreiber on
transfer entropy, and the continuous-time transfer-entropy literature for jump and
point processes.

---

## 13. Bottom-line research judgment

The project is mathematically viable and sufficiently novel for a statistics PhD if
the contribution is framed as a comparison of **statistical experiments generated by
stopped stochastic evaporation models**, not as a numerical proof of quantum gravity.
The decisive methodological contribution is the controlled comparison of filtrations,
likelihood geometry, and information flow under discrete and continuous latent laws.

The raw requested models are appropriate as the first phenomenological experiment.
The moment-matched diffusion, common detector, causal filtering requirement, and
common-parameter Fisher pullback are necessary for the thesis's central conclusion to
be identifiable and defensible.
