#!/usr/bin/env python3
"""Part 3: run the centralized information engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from black_hole_information.cli import main as cli_main  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.json"))
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--output", default=str(ROOT / "artifacts" / "part3" / "information_summary.json")
    )
    args = parser.parse_args()
    return cli_main(
        [
            "run",
            "--config",
            args.config,
            "--replicates",
            str(args.replicates),
            "--seed",
            str(args.seed),
            "--output",
            args.output,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
