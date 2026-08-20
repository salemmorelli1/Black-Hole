#!/usr/bin/env python3
"""Part 0: validate the mathematical configuration and matching relationships."""

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
    parser.add_argument(
        "--output", default=str(ROOT / "artifacts" / "part0" / "model_validation.json")
    )
    args = parser.parse_args()
    return cli_main(["validate", "--config", args.config, "--output", args.output])


if __name__ == "__main__":
    raise SystemExit(main())
