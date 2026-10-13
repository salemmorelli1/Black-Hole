"""Command-line interface for validated research runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .artifacts import provenance, tensor_to_data, write_json_atomic
from .engine import CentralizedInformationEngine, PhysicsConfig, mean_drift_matching


@dataclass(frozen=True)
class LoadedExperiment:
    config: PhysicsConfig
    theta_values: dict[str, float]
    document: dict[str, Any]
    source_sha256: str


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = value
    return result


def load_experiment_snapshot(path: Path) -> LoadedExperiment:
    """Hash the exact bytes that are parsed, before any simulation can run."""
    source = path.read_bytes()
    raw = json.loads(source, object_pairs_hook=_unique_object)

    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a JSON object")
    theta_raw = raw.get("theta")
    if not isinstance(theta_raw, dict):
        raise ValueError("configuration must contain a theta object")
    expected_theta = {"M0", "gamma", "alpha"}
    if set(theta_raw) != expected_theta:
        raise ValueError("theta must contain exactly M0, gamma, and alpha")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in theta_raw.values()
    ):
        raise ValueError("theta values must be real JSON numbers, not booleans or strings")

    config_raw = {key: value for key, value in raw.items() if key != "theta"}
    config = PhysicsConfig(**config_raw)
    config.validate()
    try:
        theta_values = {
            "M0": float(theta_raw["M0"]),
            "gamma": float(theta_raw["gamma"]),
            "alpha": float(theta_raw["alpha"]),
        }
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("theta values must be real numbers") from exc
    if not all(isfinite(value) for value in theta_values.values()):
        raise ValueError("theta values must be finite")
    if theta_values["M0"] <= config.m_floor:
        raise ValueError("theta.M0 must exceed m_floor")
    if theta_values["gamma"] <= 0.0 or theta_values["alpha"] <= 0.0:
        raise ValueError("theta.gamma and theta.alpha must be positive")
    return LoadedExperiment(config, theta_values, raw, hashlib.sha256(source).hexdigest())


def load_experiment_config(path: Path) -> tuple[PhysicsConfig, dict[str, float], dict[str, Any]]:
    """Backward-compatible three-value view of an exact configuration snapshot."""
    experiment = load_experiment_snapshot(path)
    return experiment.config, experiment.theta_values, experiment.document


def make_theta(values: dict[str, float], requires_grad: bool) -> torch.Tensor:
    if set(values) != {"M0", "gamma", "alpha"}:
        raise ValueError("values must contain exactly M0, gamma, and alpha")
    if not all(isfinite(value) for value in values.values()):
        raise ValueError("theta values must be finite")
    return torch.tensor(
        [values["M0"], values["gamma"], values["alpha"]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )


def validation_payload(config_path: Path) -> dict[str, Any]:
    experiment = load_experiment_snapshot(config_path)
    config, theta_values, raw = experiment.config, experiment.theta_values, experiment.document
    theta = make_theta(theta_values, requires_grad=False)
    matching = mean_drift_matching(config, theta)
    alpha_from_a = theta[1] * matching["alpha_per_gamma"]
    return {
        "schema_version": "2.0",
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
        "matched_control_drift": "gamma * hawking_kappa * E[X], independent of requested alpha",
        "matched_control_scope": "interior local moments; jump clipping and stopped Euler boundaries are different",
        "provenance": provenance(
            config_path, seed=0, replicates=0, config_sha256=experiment.source_sha256
        ),
        "warnings": [
            "The requested B diffusion is not globally moment-matched to A.",
            "Transfer entropy requires causal observation-history predictors.",
            "All paths are stopped at m_floor; no M=0 claim is made.",
            "Noisy initial calibration does not by itself establish regular M0 Fisher information at moving cutoff support.",
        ],
    }


def command_validate(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    payload = validation_payload(config_path)
    if args.output:
        output_path = Path(args.output)
        write_json_atomic(output_path, payload)
        print(f"Wrote {output_path}")
    print(json.dumps(tensor_to_data(payload), indent=2, sort_keys=True))
    return 0


def command_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    experiment = load_experiment_snapshot(config_path)
    config, theta_values, raw = experiment.config, experiment.theta_values, experiment.document
    run_provenance = provenance(
        config_path, int(args.seed), int(args.replicates), config_sha256=experiment.source_sha256
    )
    theta = make_theta(theta_values, requires_grad=True)
    result = CentralizedInformationEngine(config).evaluate(
        theta=theta,
        replicates=int(args.replicates),
        seed=int(args.seed),
    )
    output_path = Path(args.output)
    payload = {
        "schema_version": "2.0",
        "package_version": __version__,
        "status": "PASS",
        "config": raw,
        "theta_order": ["M0", "gamma", "alpha"],
        "theta": theta,
        "result": result,
        "provenance": run_provenance,
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
