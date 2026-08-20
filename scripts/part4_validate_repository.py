#!/usr/bin/env python3
"""Part 4: run the repository's numerical validation suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from black_hole_information.artifacts import write_json_atomic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default=str(ROOT / "artifacts" / "part4" / "validation_summary.json")
    )
    args = parser.parse_args()

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    status = "PASS" if completed.returncode == 0 else "FAIL"
    payload = {
        "schema_version": "1.0",
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    output = Path(args.output)
    write_json_atomic(output, payload)
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    print(f"Wrote {output}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
