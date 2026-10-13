# Stopped-experiment and evidence audit

Audit date: October 12, 2026 (America/New_York). Baseline:
`c4739b0aaa6afb583e9af3570f91dabf717ee474`. Proposed package: 0.2.0.

The September interval-integration correction remains in place. This audit
preserves the requested A/B laws, positive cutoff, finite Planck mixture,
fixed-data score convention, common-coordinate pullback, and the distinction
between open-loop contrasts and transfer entropy.

## Material findings and corrections

1. **The packet information calculation continued after absorption.** The
   simulator stops, but the evaluator divided gamma by the clamped mass at
   later grid points. With gamma=4 and floor=0.25, an absorbed path had intensity
   16 instead of 0. Intensity now includes the active-state indicator, and both
   entropy generators vanish at the cutoff.

2. **The packet entropy expression ignored censored emissions.** Its interior
   formula could predict positive entropy drift close to the cutoff, despite
   every stopped packet reducing mass. The stopped generator now uses
   `E[min(X,u)]` and `E[min(X,u)^2]`, where `u=(M-floor)/T_H(M)`, including the
   terminal survival atom. Independent Gauss-Legendre integration of the
   survival-function identities checks both capped moments.

3. **B_star matched variance but not drift.** It used the independent requested
   alpha. Its drift now uses `alpha_star=gamma*kappa*E[X]`; changing requested
   alpha leaves B_star paths unchanged at the same seed. The requested B still
   uses its requested alpha and sigma/M scaling. B_star is an *interior*
   local-moment control: capped jumps and stopped Euler proposals have different
   boundary behavior, so global matching at the cutoff is not claimed.

4. **Boundary likelihood probabilities were floored at epsilon.** This removed
   valid rare-event information and gradients. In a one-step example with
   M0=1, floor=0.25, alpha=1, sigma=0.01 and dt=0.01, the old boundary ratio was
   effectively 0; stable log-CDF evaluation gives approximately 7450.0134.
   The implementation now uses `torch.special.log_ndtr`. A separate example
   with standard-normal thresholds -10 and -9.9 is checked against `math.erfc`.
   These are censored Euler-proposal probabilities, not exact between-grid
   first-passage probabilities of the continuous diffusion.

5. **A Girsanov ratio was used as the full joint diffusion score.** The reference
   distribution depends on the reconstructed starting mass and, for B_star,
   the diffusion parameter. Dropping its normalized density loses score terms.
   The evaluator now uses a normalized finite-grid likelihood of regular
   increments, censored hit flags, and the noisy initial calibration. Perfect
   simulated starting mass is metadata, not a second observation of unknown
   M0. At a one-step mean increment with y0=M0, the normalized M0 score includes
   `1/M0` from the scale; the old ratio-based score missed it. Normal densities
   and central finite differences independently verify the corrected score.
   The ratio method remains available as an explicitly labeled diagnostic.

6. **Zero reduced intensity was converted into finite information.** The count
   functional now retains its mathematical limits: D(0||r)=r, D(0||0)=0, and
   D(f||0)=infinity for f>0. The engine rejects an unsupported predictor before
   publishing a finite contrast/gradient. Near-equal rates use log1p, and a
   stable prefix/suffix leave-one-out mean avoids cancellation that could
   manufacture a zero predictor. Epsilon no longer changes the sampled
   waiting-time or mark laws; only the floating dtype's representable lower
   limit protects a machine-zero uniform draw.

7. **Evidence could identify a different configuration from the one used.** The
   loader hashes the exact bytes it parses, before simulation. The runner
   freezes one input snapshot for all stages, resolves caller-relative paths
   before changing working directory, and verifies each artifact's run ID,
   input hash, implementation hashes, and successful status. A pipeline
   manifest records failures instead of accepting stale PASS files. Explicitly
   skipping tests gives `SIMULATION_ONLY`, not a validated pipeline PASS.

8. **Repository validation could pass without running tests.** Part 4 now
   records actual reported test counts and fails for zero/missing/all-skipped
   suites, even when unittest's process exit code is zero. Source verification,
   Linux/Windows CI, pipeline execution, compilation, and package build gates
   are included in the PR checks.

9. **The browser demonstration overstated its scores.** Its weighted illustration
   scores are not the Python information functionals. Labels and captions now
   explicitly prevent interpreting their sign as evidence for the hypothesis.
   Its B trajectory also stays absorbed instead of restarting from the floor.
   The two HTML copies remain identical. The August report is identified as
   historical, and its link works independently of the Pages source folder.

The finite-mixture approximation is also described correctly: truncating and
renormalizing weights proportional to k^-3 has an algebraic O(K^-2) tail error,
not an exponentially small global normalization error. A conservative mixture
total-variation bound, `1/(2*K^2)`, is included in the matching metadata.

## Statistical interpretation that remains essential

- Complete-data score outer products are not observed-detector Fisher
  information. A noisy initial calibration does not by itself establish
  regularity of M0 when cutoff support moves with that parameter. The result
  metadata therefore qualifies the matrices as fixed-data score geometry and
  does not assert a regular Fisher metric or a Cramer-Rao bound.
- The A and requested B pullbacks can be evaluated at different alpha_eff
  coordinates. Both points are recorded. A pullback does not automatically
  align model parameter points or detector experiments.
- The Gaussian detector continues to read the *stopped* mass through the
  written h(M)=gain/M^2 channel. Constant readout at the floor is not ongoing
  evaporation. Packet emissions and entropy drift stop; the declared Gaussian
  readout definition is preserved.
- Supplying a tensor is not proof that a predictor is causal. Such predictors
  are labeled unverified. Default leave-one-out predictors remain open-loop.
- No Monte Carlo confidence interval is claimed. Leave-one-out predictors
  couple replicates; treating their contrasts as independent observations
  would omit uncertainty from the fitted predictor. Per-path values are saved
  to support a future appropriate inference procedure.
- A matched/common-detector experiment, causal filters, uncertainty analysis,
  and convergence studies are still needed for a thesis-level transfer-entropy
  conclusion. This audit does not constitute evidence for quantum gravity.

## Reproduction and release identity

`SOURCE_SHA256SUMS.json` and `verify_source.py` bind the delivered text files.
New checkouts use LF endings on Windows as well as Linux. Each numerical
artifact records the captured input hash and the actual Python implementation
hashes, along with the runtime, seed, replicate count, and run ID.

The 17 initial stopped-experiment regressions failed against the unmodified
baseline and passed after correction. Additional tests cover sampling-law
independence from epsilon, stable leave-one-out prediction, frozen inputs,
ambiguous JSON, stale stage evidence, nonempty validation, source tampering,
and publication-copy consistency. Real-browser tests check the labels and
absorbing visual trajectory separately from the physics engine.

Use the PR's immutable tested commit and GitHub checks as the release evidence.
Generated pipeline results remain in ignored `artifacts/`; the historical PDF
is not presented as a validation report for version 0.2.0.

Primary implementation/theory references:

- [PyTorch special functions: log_ndtr and gammainc](https://docs.pytorch.org/docs/stable/special.html)
- [Spinney, Prokopenko and Lizier: continuous-time transfer entropy](https://arxiv.org/abs/1610.08192)

The stopped-generator and normalized Euler formulas above are derived directly
from this repository's declared models. They are implementation corrections,
not new empirical claims about physical black holes.
