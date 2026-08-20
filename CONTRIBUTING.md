# Contributing

Contributions should preserve the distinction between physical assumptions,
statistical estimands, and numerical approximations.

1. Create a focused branch.
2. Add or update tests for every mathematical or numerical change.
3. Run `python -m unittest discover -s tests -v`.
4. Run `ruff check .`.
5. Document changes to likelihoods, filtrations, stopping rules, or parameter maps.
6. Do not label an open-loop information contrast as transfer entropy.
7. Do not remove the positive mass cutoff without a replacement physical model.

Generated artifacts, model checkpoints, and large arrays should not be committed.
