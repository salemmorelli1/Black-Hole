"""Artifact serialization and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def tensor_to_data(value: Any) -> Any:
    """Recursively convert tensors and paths into JSON-compatible objects."""

    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.ndim == 0 else detached.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): tensor_to_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_data(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(config_path: Path, seed: int, replicates: int) -> dict[str, Any]:
    """Create an auditable run manifest."""

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "seed": int(seed),
        "replicates": int(replicates),
        "torch_version": torch.__version__,
        "default_dtype": str(torch.get_default_dtype()),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so interrupted runs cannot leave partial artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = tensor_to_data(payload)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(serializable, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


__all__ = ["provenance", "sha256_file", "tensor_to_data", "write_json_atomic"]
