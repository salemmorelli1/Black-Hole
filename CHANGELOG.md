# Changelog


## 0.2.0 — 2026-10-12

- Stop packet rates and entropy generators at absorption; use capped Planck
  moments for the stopped jump entropy generator.
- Match B_star's interior drift to A and retain the requested B model.
- Use normalized finite-Euler increment likelihoods for diffusion scores and
  stable log-CDFs for censored proposals. Validate fixed-data path support.
- Preserve zero-intensity information limits, reject unsupported predictors,
  and stabilize leave-one-out means. Qualify formal score geometry and causal
  predictor status; record both common-coordinate evaluation points.
- Freeze exact pipeline inputs, bind fresh stage evidence to a run ID and
  implementation hashes, and reject empty validation suites.
- Identify browser scores and the historical report accurately, preserve
  absorbing visual trajectories, and add Windows/Edge and source-byte checks.

All notable changes are documented here.

## Unreleased

### Fixed

- Integrate information rates over the `n_steps` physical intervals instead of
  counting the terminal grid state as an extra interval.
- Reject non-finite configurations, nonphysical parameters, invalid Planck-mark
  support, negative intensities, and zero observation noise before calculation.
- Reject non-standard JSON `NaN`/`Infinity` values without replacing an existing
  artifact.
- Pin direct CI dependencies and GitHub Actions revisions, and expand the gate to
  include type checking, compilation, and the repository validation entry point.

## 0.1.0 - 2026-08-20

### Added

- Stopped marked point-process simulator for discrete Hawking packets.
- Stopped Euler-Maruyama simulator for the requested continuous diffusion.
- Moment-matched diffusion control with \(M^{-3/2}\) noise scaling.
- Planck number-spectrum mixture sampler and analytic moment checks.
- Complete-path marked point-process likelihood.
- Girsanov path likelihood with a censored boundary contribution.
- Bekenstein-Hawking entropy-drift diagnostics.
- Autograd likelihood scores and empirical Fisher matrices.
- Common-parameter Fisher pullback and matrix residual.
- Point-process and Gaussian-channel information functionals.
- Canonical five-stage research pipeline, CLI, CI, documentation, and tests.
