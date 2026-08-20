# Architecture and artifact contracts

## Dependency graph

```text
configs/baseline.json
        |
        v
Part 0: mathematical validation
        |
        +-------------------+
        |                   |
        v                   v
Part 1: Paradigm A     Part 2: Paradigm B and B*
        |                   |
        +---------+---------+
                  v
        Part 3: information engine
                  |
                  v
        Part 4: repository validation
```

The scientific implementation lives in `src/black_hole_information`. Numbered
scripts are thin, auditable orchestration layers. This avoids duplicating mathematical
logic while preserving an explicit production order.

## Part 0 contract

Input:

- `configs/baseline.json`

Output:

- `artifacts/part0/model_validation.json`

Required fields include Planck moments, the drift-matching map, the requested and
matched noise scalings, configuration hash, and explicit warnings.

## Part 1 contract

Input:

- validated physical configuration;
- parameter vector ordered as `(M0, gamma, alpha)`;
- reproducible random seed.

Output:

- `artifacts/part1/paradigm_a_summary.json`

The summary records event-count moments, final-mass moments, boundary absorption, and
run provenance. Full paths are intentionally not committed by default.

## Part 2 contract

Input:

- the same configuration, parameters, replicate count, and seed policy as Part 1.

Output:

- `artifacts/part2/paradigm_b_summary.json`

The output separates the requested diffusion B from the moment-matched control
`B_star`.

## Part 3 contract

Output:

- `artifacts/part3/information_summary.json`

Required quantities:

- native Fisher matrices for A and B;
- common-parameter Fisher pullbacks;
- matrix residual `B - A`;
- integrated information values in nats;
- scalar information residual;
- conditional pathwise gradient of the scalar residual;
- entropy-rate summaries;
- predictor-status labels specifying whether transfer-entropy conditions hold.

## Part 4 contract

Output:

- `artifacts/part4/validation_summary.json`

A failed unit test produces a nonzero process exit code. The canonical runner stops
immediately and does not claim a successful pipeline.

## Derivative semantics

The repository recognizes three separate derivatives:

1. **Fixed-data likelihood score:** defines Fisher information.
2. **Pathwise simulation derivative:** differentiates through base random variables
   conditional on a realized jump topology.
3. **Topology derivative:** accounts for parameter-dependent changes in event count or
   boundary status and generally requires a score-function or weak-derivative method.

The engine implements the first two and labels the limitation on the third.

## Reproducibility policy

Every artifact contains:

- UTC creation time;
- source configuration path and SHA-256 digest;
- random seed;
- replicate count;
- PyTorch version;
- default tensor dtype.

All scientific calculations use float64 by default.
