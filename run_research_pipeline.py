#!/usr/bin/env python3
"""Canonical Part 0 -> Part 4 research runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from black_hole_information.artifacts import source_hashes, write_json_atomic  # noqa: E402


def freeze_configuration(path: Path, directory: Path) -> tuple[Path, str]:
    """Keep one exact input snapshot for every stage, even if the input is edited."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest}.json"
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError("the existing configuration snapshot does not match its hash")
        return target.resolve(), digest
    handle, temporary_name = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target.resolve(), digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.json"))
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args(argv)
    if args.replicates < 2:
        parser.error("--replicates must be at least 2 for the information engine")
    input_path = Path(args.config).expanduser().resolve()
    output_root = args.output_root.resolve()
    snapshot, config_digest = freeze_configuration(input_path, output_root / "input-configs")
    run_id = uuid.uuid4().hex
    environment = dict(os.environ)
    environment["BHID_RUN_ID"] = run_id
    expected_sources = source_hashes()
    manifest: dict = {
        "schema_version": "2.0",
        "status": "RUNNING",
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_config_path": str(input_path),
        "config_snapshot_path": str(snapshot),
        "config_sha256": config_digest,
        "source_sha256": expected_sources,
        "seed": args.seed,
        "replicates": args.replicates,
        "tests": "NOT_RUN" if args.skip_tests else "PENDING",
        "stages": [],
    }
    manifest_path = output_root / "pipeline_manifest.json"
    write_json_atomic(manifest_path, manifest)

    shared = ["--config", str(snapshot)]
    simulation = ["--replicates", str(args.replicates), "--seed", str(args.seed)]
    stages = [
        (
            "Part 0",
            "part0/model_validation.json",
            [sys.executable, "scripts/part0_validate_model.py", *shared],
        ),
        (
            "Part 1",
            "part1/paradigm_a_summary.json",
            [sys.executable, "scripts/part1_simulate_paradigm_a.py", *shared, *simulation],
        ),
        (
            "Part 2",
            "part2/paradigm_b_summary.json",
            [sys.executable, "scripts/part2_simulate_paradigm_b.py", *shared, *simulation],
        ),
        (
            "Part 3",
            "part3/information_summary.json",
            [sys.executable, "scripts/part3_information_engine.py", *shared, *simulation],
        ),
    ]
    if not args.skip_tests:
        stages.append(
            (
                "Part 4",
                "part4/validation_summary.json",
                [sys.executable, "scripts/part4_validate_repository.py"],
            )
        )

    for label, relative, command in stages:
        output = output_root / relative
        command = [*command, "--output", str(output)]
        print(f"\n[{label}] {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        failure = None
        artifact_digest = None
        if completed.returncode == 0:
            try:
                data = output.read_bytes()
                payload = json.loads(data)
                metadata = payload if label == "Part 4" else payload.get("provenance", {})
                if (
                    payload.get("status") != "PASS"
                    or metadata.get("run_id") != run_id
                    or metadata.get("source_sha256") != expected_sources
                ):
                    raise ValueError(
                        "artifact is not PASS evidence for this run and implementation"
                    )
                if label != "Part 4" and metadata.get("config_sha256") != config_digest:
                    raise ValueError("artifact configuration differs from the captured input")
                artifact_digest = hashlib.sha256(data).hexdigest()
            except (OSError, ValueError, AttributeError) as exc:
                failure = str(exc)
        else:
            failure = f"exit code {completed.returncode}"
        manifest["stages"].append(
            {
                "stage": label,
                "status": "FAIL" if failure else "PASS",
                "output": str(output),
                "sha256": artifact_digest,
                "failure": failure,
            }
        )
        if failure:
            manifest["status"] = "FAIL"
            manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
            write_json_atomic(manifest_path, manifest)
            print(f"[{label}] FAILED: {failure}", file=sys.stderr)
            return completed.returncode or 1
        if label == "Part 4":
            manifest["tests"] = "PASS"
        write_json_atomic(manifest_path, manifest)
        print(f"[{label}] PASS", flush=True)
    manifest["status"] = "SIMULATION_ONLY" if args.skip_tests else "PASS"
    manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json_atomic(manifest_path, manifest)
    print(f"Pipeline evidence: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
