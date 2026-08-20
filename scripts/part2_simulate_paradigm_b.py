#!/usr/bin/env python3
"""Part 2: generate and summarize requested and moment-matched diffusions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from black_hole_information.artifacts import provenance, write_json_atomic  # noqa: E402
from black_hole_information.cli import load_experiment_config, make_theta  # noqa: E402
from black_hole_information.engine import ContinuousMassDiffusion  # noqa: E402


def summarize(model, theta, generator, replicates):
    final_masses = []
    hit_boundary = []
    for _ in range(replicates):
        path = model.simulate(theta, generator)
        final_masses.append(path.masses[-1])
        hit_boundary.append(float(path.hit_transition.any()))
    masses = torch.stack(final_masses)
    return {
        "mean_final_mass": masses.mean(),
        "final_mass_sd": masses.std(unbiased=False),
        "boundary_hit_fraction": sum(hit_boundary) / len(hit_boundary),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.json"))
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--output", default=str(ROOT / "artifacts" / "part2" / "paradigm_b_summary.json")
    )
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)

    config_path = Path(args.config)
    config, theta_values, raw = load_experiment_config(config_path)
    theta = make_theta(theta_values, requires_grad=False)
    requested = ContinuousMassDiffusion(config, moment_matched=False)
    matched = ContinuousMassDiffusion(config, moment_matched=True)

    requested_generator = torch.Generator().manual_seed(args.seed)
    matched_generator = torch.Generator().manual_seed(args.seed)
    payload = {
        "schema_version": "1.0",
        "status": "PASS",
        "paradigm": "B_continuous_mass_diffusion",
        "config": raw,
        "requested_B": summarize(requested, theta, requested_generator, args.replicates),
        "moment_matched_B_star": summarize(matched, theta, matched_generator, args.replicates),
        "provenance": provenance(config_path, args.seed, args.replicates),
    }
    output = Path(args.output)
    write_json_atomic(output, payload)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
