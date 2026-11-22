"""Atomic checkpoints and config fingerprints for A1 runs.

A checkpoint is written to ``<name>.tmp`` in the same folder, flushed to disk,
and then moved over the old file with ``os.replace``. An interruption at any
point leaves either the previous complete checkpoint or the new complete one,
never a half-written file. This works on Windows and Linux.

Checkpoints hold only tensors and plain Python containers, so they load with
``torch.load(weights_only=True)``: loading never executes pickled code.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Union

import torch

__all__ = [
    "CHECKPOINT_FORMAT",
    "config_hash",
    "save_checkpoint",
    "load_checkpoint",
    "atomic_write_text",
]

CHECKPOINT_FORMAT = 1
PathLike = Union[str, Path]


def config_hash(cfg: Dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON form of a config dict."""
    text = json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_with_retry(src: Path, dst: Path, attempts: int = 5) -> None:
    # On Windows, os.replace can fail briefly if another process (an antivirus
    # scanner, a file indexer) has the target open. Retry a few times.
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * (2 ** i))


def _fsync(path: Path) -> None:
    with open(path, "rb+") as fh:
        os.fsync(fh.fileno())


def save_checkpoint(path: PathLike, state: Dict[str, Any]) -> None:
    """Atomically write ``state`` (plus a format tag) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = {"format": CHECKPOINT_FORMAT, **state}
    try:
        torch.save(payload, tmp)
        _fsync(tmp)
        _replace_with_retry(tmp, path)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def load_checkpoint(path: PathLike, expected_config_hash: str | None = None) -> Dict[str, Any]:
    """Load a checkpoint safely (weights_only) and check its format and config."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not an A1 checkpoint (format {CHECKPOINT_FORMAT})")
    if expected_config_hash is not None and state.get("config_hash") != expected_config_hash:
        raise ValueError(
            f"checkpoint {path} was written for a different config "
            f"(saved {state.get('config_hash', '?')[:12]}, current {expected_config_hash[:12]}); "
            "refusing to resume. Use a new run folder or delete the old checkpoint."
        )
    return state


def atomic_write_text(path: PathLike, text: str) -> None:
    """Write a text file atomically (same temp-then-replace scheme)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
