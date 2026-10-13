"""Configuration custody and nonempty validation evidence regressions."""

import contextlib
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_research_pipeline as pipeline
from black_hole_information.artifacts import provenance, source_hashes
from black_hole_information.cli import load_experiment_config, load_experiment_snapshot
from scripts import part4_validate_repository as validation


class ConfigurationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.path = self.root / "input.json"
        self.payload = {
            "horizon": 0.02,
            "dt": 0.01,
            "theta": {"M0": 4.0, "gamma": 2.0, "alpha": 0.2},
        }

    def test_hash_binds_the_bytes_loaded_even_after_the_file_changes(self):
        original = json.dumps(self.payload, indent=3).encode()
        self.path.write_bytes(original)
        experiment = load_experiment_snapshot(self.path)
        self.payload["theta"]["alpha"] = 0.5
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")
        record = provenance(self.path, 3, 4, config_sha256=experiment.source_sha256)
        self.assertEqual(record["config_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(experiment.theta_values["alpha"], 0.2)
        self.assertEqual(
            set(record["source_sha256"]), {"__init__.py", "artifacts.py", "cli.py", "engine.py"}
        )

    def test_ambiguous_or_boolean_configuration_values_are_rejected(self):
        for source in (
            '{"theta":{"M0":4,"gamma":2,"alpha":0.2,"alpha":0.5}}',
            '{"theta":{"M0":4,"gamma":true,"alpha":0.2}}',
            '{"theta":{"M0":4,"gamma":"2","alpha":0.2}}',
        ):
            self.path.write_text(source, encoding="utf-8")
            with self.subTest(source=source), self.assertRaises(ValueError):
                load_experiment_config(self.path)

    def test_frozen_snapshot_is_exact_and_survives_original_edits(self):
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")
        original = self.path.read_bytes()
        target, digest = pipeline.freeze_configuration(self.path, self.root / "snapshots")
        self.path.write_text('{"changed": true}', encoding="utf-8")
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(digest, hashlib.sha256(original).hexdigest())

    def test_corrupted_existing_snapshot_is_rejected_without_overwriting(self):
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")
        target, _ = pipeline.freeze_configuration(self.path, self.root / "snapshots")
        target.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            pipeline.freeze_configuration(self.path, self.root / "snapshots")
        self.assertEqual(target.read_bytes(), b"corrupted")

    def test_pipeline_requires_fresh_stage_evidence_for_the_same_input(self):
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")
        output_root = self.root / "outputs"
        calls = []
        captured_inputs = []

        def stage(command, *, cwd, env, check):
            calls.append(command)
            selected = Path(command[command.index("--config") + 1])
            captured_inputs.append(selected.read_bytes())
            original = selected.read_bytes()
            # Editing the original after Part 0 must not change subsequent stages.
            self.path.write_text('{"changed": true}', encoding="utf-8")
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "provenance": {
                            "run_id": env["BHID_RUN_ID"],
                            "source_sha256": source_hashes(),
                            "config_sha256": hashlib.sha256(original).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch.object(pipeline.subprocess, "run", side_effect=stage),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                pipeline.main(
                    [
                        "--config",
                        str(self.path),
                        "--output-root",
                        str(output_root),
                        "--replicates",
                        "2",
                        "--skip-tests",
                    ]
                ),
                0,
            )
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(data == captured_inputs[0] for data in captured_inputs))
        manifest = json.loads((output_root / "pipeline_manifest.json").read_text())
        self.assertEqual(manifest["status"], "SIMULATION_ONLY")
        self.assertEqual(manifest["tests"], "NOT_RUN")

    def test_pipeline_cannot_accept_an_old_success_artifact(self):
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")
        output_root = self.root / "outputs"
        output = output_root / "part0/model_validation.json"
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps({"status": "PASS", "provenance": {"run_id": "old-run"}}))
        with (
            mock.patch.object(
                pipeline.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                pipeline.main(
                    [
                        "--config",
                        str(self.path),
                        "--output-root",
                        str(output_root),
                        "--replicates",
                        "2",
                    ]
                ),
                1,
            )
        manifest = json.loads((output_root / "pipeline_manifest.json").read_text())
        self.assertEqual(manifest["status"], "FAIL")
        self.assertEqual(len(manifest["stages"]), 1)


class NonemptyValidationTests(unittest.TestCase):
    def test_zero_missing_or_all_skipped_tests_cannot_report_pass(self):
        for log in ("Ran 0 tests in 0.0s\nOK", "OK", "Ran 2 tests in 0.0s\nOK (skipped=2)"):
            with self.subTest(log=log), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "validation.json"
                completed = subprocess.CompletedProcess([], 0, stdout="", stderr=log)
                with (
                    mock.patch.object(validation.subprocess, "run", return_value=completed),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(validation.main(["--output", str(target)]), 1)
                self.assertEqual(json.loads(target.read_text())["status"], "FAIL")

    def test_nonempty_suite_keeps_actual_counts_and_failures(self):
        for code, summary in ((0, "OK"), (1, "FAILED (failures=1)")):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "validation.json"
                completed = subprocess.CompletedProcess(
                    [], code, stdout="", stderr="Ran 3 tests in 0.1s\n" + summary
                )
                with (
                    mock.patch.object(validation.subprocess, "run", return_value=completed),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(validation.main(["--output", str(target)]), code)
                payload = json.loads(target.read_text())
                self.assertEqual(payload["tests_run"], 3)
                self.assertEqual(payload["status"], "PASS" if code == 0 else "FAIL")


if __name__ == "__main__":
    unittest.main()
