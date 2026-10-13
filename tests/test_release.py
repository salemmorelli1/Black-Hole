"""Delivered source verification and publication-copy consistency."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import verify_source

ROOT = Path(__file__).resolve().parents[1]


class SourceVerificationTests(unittest.TestCase):
    def test_changed_source_is_rejected_without_modifying_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "example.py"
            source.write_bytes(b"tested source\n")
            manifest = {"example.py": hashlib.sha256(source.read_bytes()).hexdigest()}
            (root / "SOURCE_SHA256SUMS.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_source.verify(root), 1)
            source.write_bytes(b"local edit\n")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_source.verify(root)
            self.assertEqual(source.read_bytes(), b"local edit\n")

    def test_manifest_cannot_escape_the_delivery_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "SOURCE_SHA256SUMS.json").write_text(json.dumps({"../outside.py": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                verify_source.verify(root)

    def test_publication_copies_are_identical_and_labels_are_explicit(self):
        content = (ROOT / "index.html").read_bytes()
        self.assertEqual(content, (ROOT / "docs/index.html").read_bytes())
        text = content.decode()
        self.assertIn("Their sign supplies no evidence for the research hypothesis", text)
        self.assertIn("tables predate the version 0.2.0 numerical audit", text)
        self.assertIn("not a measured information residual", text)


if __name__ == "__main__":
    unittest.main()
