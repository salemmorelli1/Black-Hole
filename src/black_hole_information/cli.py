"""Command-line interface for validated research runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .artifacts import provenance, tensor_to_data, write_json_atomic
from .engine import CentralizedInformationEngine, PhysicsConfig, mean_drift_matching


def load_experiment_config(path: Path) -> tuple[PhysicsConfig, dict[str, float], dict[str, Any]]:
    """Load and validate a JSON experiment configuration."""

    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    theta_raw = raw.pop("theta")
    config = PhysicsConfig(**raw)
    config.validate()
    theta_values = {
        "M0": float(theta_raw["M0"]),
        "gamma": float(theta_raw["gamma"]),
        "alpha": float(theta_raw["alpha"]),
    }
    if theta_values["M0"] <= config.m_floor:
        raise ValueError("theta.M0 must exceed m_floor")
    if theta_values["gamma"] <= 0.0 or theta_values["alpha"] <= 0.0:
        raise ValueError("theta.gamma and theta.alpha must be positive")
    return config, theta_values, {**raw, "theta": theta_raw}


def make_theta(values: dict[str, float], requires_grad: bool) -> torch.Tensor:
    return torch.tensor(
        [values["M0"], values["gamma"], values["alpha"]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )


def validation_payload(config_path: Path) -> dict[str, Any]:
    config, theta_values, raw = load_experiment_config(config_path)
    theta = make_theta(theta_values, requires_grad=False)
    matching = mean_drift_matching(config, theta)
    alpha_from_a = theta[1] * matching["alpha_per_gamma"]
    return {
        "schema_version": "1.0",
        "package_version": __version__,
        "status": "PASS",
        "config": raw,
        "theta_order": ["M0", "gamma", "alpha"],
        "theta": theta,
        "matching": matching,
        "alpha_implied_by_paradigm_a": alpha_from_a,
        "alpha_requested_by_paradigm_b": theta[2],
        "drift_mismatch": theta[2] - alpha_from_a,
        "required_noise_scaling_for_B_star": "M^(-3/2)",
        "requested_noise_scaling_for_B": "M^(-1)",
        "warnings": [
            "The requested B diffusion is not globally moment-matched to A.",
            "Transfer entropy requires causal observation-history predictors.",
            "All paths are stopped at m_floor; no M=0 claim is made.",
        ],
    }


def command_validate(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    payload = validation_payload(config_path)
    if args.output:
        output_path = Path(args.output)
        payload["provenance"] = provenance(config_path, seed=0, replicates=0)
        write_json_atomic(output_path, payload)
        print(f"Wrote {output_path}")
    print(json.dumps(tensor_to_data(payload), indent=2, sort_keys=True))
    return 0


def command_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config, theta_values, raw = load_experiment_config(config_path)
    theta = make_theta(theta_values, requires_grad=True)
    result = CentralizedInformationEngine(config).evaluate(
        theta=theta,
        replicates=int(args.replicates),
        seed=int(args.seed),
    )
    output_path = Path(args.output)
    payload = {
        "schema_version": "1.0",
        "package_version": __version__,
        "status": "PASS",
        "config": raw,
        "theta_order": ["M0", "gamma", "alpha"],
        "theta": theta,
        "result": result,
        "provenance": provenance(config_path, int(args.seed), int(args.replicates)),
    }
    write_json_atomic(output_path, payload)
    print(f"Wrote {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bhid",
        description="Black Hole Information Dynamics research CLI",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate model and configuration")
    validate.add_argument("--config", default="configs/baseline.json")
    validate.add_argument("--output")
    validate.set_defaults(handler=command_validate)

    run = subparsers.add_parser("run", help="run the centralized information engine")
    run.add_argument("--config", default="configs/baseline.json")
    run.add_argument("--replicates", type=int, default=64)
    run.add_argument("--seed", type=int, default=20260820)
    run.add_argument("--output", default="artifacts/part3/information_summary.json")
    run.set_defaults(handler=command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_default_dtype(torch.float64)
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
