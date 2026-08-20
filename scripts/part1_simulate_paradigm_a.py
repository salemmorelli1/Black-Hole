#!/usr/bin/env python3
"""Part 1: generate and summarize discrete packet trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from black_hole_information.artifacts import provenance, write_json_atomic  # noqa: E402
from black_hole_information.cli import load_experiment_config, make_theta  # noqa: E402
from black_hole_information.engine import PacketJumpDiffusion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.json"))
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--output", default=str(ROOT / "artifacts" / "part1" / "paradigm_a_summary.json")
    )
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)

    config_path = Path(args.config)
    config, theta_values, raw = load_experiment_config(config_path)
    theta = make_theta(theta_values, requires_grad=False)
    model = PacketJumpDiffusion(config)
    generator = torch.Generator().manual_seed(args.seed)

    event_counts = []
    final_masses = []
    absorbed = []
    for _ in range(args.replicates):
        path = model.simulate(theta, generator)
        event_counts.append(path.event_times.numel())
        final_masses.append(path.final_mass)
        absorbed.append(float(path.absorbed))

    counts = torch.tensor(event_counts, dtype=torch.float64)
    masses = torch.stack(final_masses)
    payload = {
        "schema_version": "1.0",
        "status": "PASS",
        "paradigm": "A_discrete_packet_jump_process",
        "config": raw,
        "summary": {
            "mean_event_count": counts.mean(),
            "event_count_sd": counts.std(unbiased=False),
            "mean_final_mass": masses.mean(),
            "final_mass_sd": masses.std(unbiased=False),
            "absorbed_fraction": sum(absorbed) / len(absorbed),
        },
        "provenance": provenance(config_path, args.seed, args.replicates),
    }
    output = Path(args.output)
    write_json_atomic(output, payload)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
