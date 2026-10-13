"""Verify the delivered source-byte manifest without importing the simulator."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


def verify(root: Path) -> int:
    root = root.resolve()
    manifest = json.loads((root / "SOURCE_SHA256SUMS.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("source manifest must be a nonempty mapping")
    for name, expected in manifest.items():
        if (
            not isinstance(name, str)
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ValueError("invalid source manifest entry")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing or unsafe source file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {name}")
    return len(manifest)


def main() -> int:
    try:
        print(f"Verified {verify(Path(__file__).parent)} source files.")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"Source verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
