#!/usr/bin/env python3
"""Part 4: run the repository's numerical validation suite."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from black_hole_information.artifacts import source_hashes, write_json_atomic  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default=str(ROOT / "artifacts" / "part4" / "validation_summary.json")
    )
    args = parser.parse_args(argv)

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    implementation = source_hashes()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    log = completed.stdout + completed.stderr
    reported = re.search(r"\bRan (\d+) tests?\b", log)
    tests_run = int(reported.group(1)) if reported else 0
    skipped = re.search(r"OK \(skipped=(\d+)", log)
    tests_skipped = int(skipped.group(1)) if skipped else 0
    verified = completed.returncode == 0 and tests_run > tests_skipped
    returncode = completed.returncode if completed.returncode else (0 if verified else 1)
    status = "PASS" if verified else "FAIL"
    payload = {
        "schema_version": "2.0",
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "returncode": returncode,
        "subprocess_returncode": completed.returncode,
        "tests_run": tests_run,
        "tests_skipped": tests_skipped,
        "source_sha256": implementation,
        "run_id": os.environ.get("BHID_RUN_ID"),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    output = Path(args.output)
    write_json_atomic(output, payload)
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    print(f"Wrote {output}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
