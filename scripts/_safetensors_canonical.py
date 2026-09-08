"""Deterministic safetensors header rewriting shared by asset builders."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


HASH_CHUNK_BYTES = 1024 * 1024
SAFETENSORS_HEADER_PREFIX_BYTES = 8


def _canonical_json_value(value: Any) -> Any:
    """Return a JSON value with every object ordered by its string keys."""
    if isinstance(value, Mapping):
        return {
            key: _canonical_json_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, list):
        return [_canonical_json_value(item) for item in value]
    return value


def _canonicalize_safetensors_header(path: Path | str) -> None:
    """Atomically rewrite a safetensors header in stable order.

    The stable order is ``__metadata__`` first, followed by tensor names in
    lexical order; metadata names and nested descriptor fields are also sorted.
    Payload bytes are copied in bounded chunks, preserving tensor offsets and
    avoiding a whole-artifact allocation.
    """
    path = Path(path)
    with path.open("rb") as source:
        header_size_bytes = source.read(SAFETENSORS_HEADER_PREFIX_BYTES)
        if len(header_size_bytes) != SAFETENSORS_HEADER_PREFIX_BYTES:
            raise ValueError(f"invalid safetensors header prefix: {path}")
        header_size = int.from_bytes(header_size_bytes, "little")
        raw_header = source.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"truncated safetensors header: {path}")
        try:
            header = json.loads(raw_header)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid safetensors header JSON: {path}") from error
        if not isinstance(header, Mapping):
            raise ValueError(f"safetensors header must be a JSON object: {path}")

        canonical_header: dict[str, Any] = {}
        if "__metadata__" in header:
            canonical_header["__metadata__"] = _canonical_json_value(
                header["__metadata__"]
            )
        for name in sorted(key for key in header if key != "__metadata__"):
            canonical_header[name] = _canonical_json_value(header[name])
        canonical_header_bytes = json.dumps(
            canonical_header,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        # Retaining the original padded header length keeps every tensor offset
        # unchanged.  The expansion branch remains safe for unusual valid input.
        if len(canonical_header_bytes) <= header_size:
            rewritten_header = canonical_header_bytes.ljust(header_size, b" ")
        else:
            padding = (-len(canonical_header_bytes)) % 8
            rewritten_header = canonical_header_bytes + (b" " * padding)

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temp_path = Path(temporary.name)
                temporary.write(len(rewritten_header).to_bytes(8, "little"))
                temporary.write(rewritten_header)
                shutil.copyfileobj(source, temporary, length=HASH_CHUNK_BYTES)
                temporary.flush()
                os.fsync(temporary.fileno())
                os.fchmod(
                    temporary.fileno(), stat.S_IMODE(path.stat().st_mode)
                )
            os.replace(temp_path, path)
        except BaseException:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise
