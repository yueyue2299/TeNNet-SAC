"""Strict loading for the deterministic tuned gamma-ensemble asset."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .models.GammaEnsemble import GammaEnsemble


BUNDLE_FILENAME = "gamma-ensemble-v1.safetensors"
BUNDLE_TENSOR_COUNT = 93
BUNDLE_TENSOR_BYTES = 5_202_560
BUNDLE_METADATA = {
    "asset_name": "gamma-tuned-ensemble",
    "format_version": "1",
    "member_count": "10",
    "shared_tensor_count": "33",
    "head_tensor_count": "60",
    "source_manifest_bundle_version": "1.0.0",
}
HASH_CHUNK_BYTES = 1024 * 1024


class GammaEnsembleLoadError(ValueError):
    """Raised when a gamma-ensemble asset fails an integrity check."""


def _is_lower_hex_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicate_names(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors header key: {key}")
        result[key] = value
    return result


def _sha256_file(path_or_stream: Path | Any) -> str:
    """Hash one path or already-open binary stream without changing its position."""
    digest = hashlib.sha256()
    if isinstance(path_or_stream, (str, os.PathLike)):
        with Path(path_or_stream).open("rb") as stream:
            for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    else:
        stream = path_or_stream
        position = stream.tell()
        try:
            stream.seek(0)
            for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        finally:
            stream.seek(position)
    return digest.hexdigest()


def _read_regular_file_bytes(path: Path, description: str) -> bytes:
    """Read one immutable descriptor so later pathname replacement is harmless."""
    entry_stat = path.lstat()
    if stat.S_ISLNK(entry_stat.st_mode):
        raise ValueError(f"{description} must not be a symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"{description} must not be a symlink: {path}") from error
        raise
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError(f"{description} must be a regular file: {path}")
        if (entry_stat.st_dev, entry_stat.st_ino) != (
            opened_stat.st_dev,
            opened_stat.st_ino,
        ):
            raise ValueError(f"{description} changed while opening: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return b"".join(
                iter(lambda: stream.read(HASH_CHUNK_BYTES), b"")
            )
    finally:
        os.close(descriptor)


def read_safetensors_header(file_bytes: bytes) -> Mapping[str, Any]:
    """Parse a raw safetensors header while rejecting duplicate JSON names."""
    if len(file_bytes) < 8:
        raise ValueError("invalid safetensors header prefix")
    header_size = int.from_bytes(file_bytes[:8], "little")
    raw_header = file_bytes[8 : 8 + header_size]
    if len(raw_header) != header_size:
        raise ValueError("truncated safetensors header")
    try:
        header = json.loads(raw_header, object_pairs_hook=_reject_duplicate_names)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GammaEnsembleLoadError("invalid safetensors header") from error
    if not isinstance(header, Mapping):
        raise ValueError("safetensors header must be a JSON object")
    return header


def _safetensors_dtype(dtype: torch.dtype) -> str:
    dtypes = {
        torch.float32: "F32",
        torch.float64: "F64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }
    try:
        return dtypes[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported expected tensor dtype: {dtype}") from error


def validate_bundle_header(
    header: Mapping[str, Any],
    expected: Mapping[str, torch.Tensor],
    *,
    payload_bytes: int | None = None,
) -> None:
    metadata = header.get("__metadata__")
    if metadata != BUNDLE_METADATA:
        raise ValueError("metadata mismatch")

    tensor_names = {name for name in header if name != "__metadata__"}
    expected_names = set(expected)
    if tensor_names != expected_names:
        missing = sorted(expected_names - tensor_names)
        unexpected = sorted(tensor_names - expected_names)
        raise ValueError(
            f"tensor names mismatch: missing {missing}, unexpected {unexpected}"
        )
    if len(tensor_names) != BUNDLE_TENSOR_COUNT:
        raise ValueError(
            f"tensor count mismatch: expected {BUNDLE_TENSOR_COUNT}, got {len(tensor_names)}"
        )

    spans: list[tuple[int, int, str]] = []
    for name in sorted(expected):
        descriptor = header[name]
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"invalid tensor descriptor: {name}")
        if set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"tensor descriptor keys mismatch: {name}")
        expected_dtype = _safetensors_dtype(expected[name].dtype)
        if descriptor.get("dtype") != expected_dtype:
            raise ValueError(
                f"tensor dtype mismatch: {name}; expected {expected_dtype}, "
                f"got {descriptor.get('dtype')}"
            )
        if descriptor.get("shape") != list(expected[name].shape):
            raise ValueError(
                f"tensor shape mismatch: {name}; expected {list(expected[name].shape)}, "
                f"got {descriptor.get('shape')}"
            )
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or type(offsets[0]) is not int
            or type(offsets[1]) is not int
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"invalid tensor data offsets: {name}")
        expected_tensor_bytes = expected[name].numel() * expected[name].element_size()
        actual_tensor_bytes = offsets[1] - offsets[0]
        if actual_tensor_bytes != expected_tensor_bytes:
            raise ValueError(
                f"tensor bytes mismatch: {name}; expected {expected_tensor_bytes}, "
                f"got {actual_tensor_bytes}"
            )
        spans.append((offsets[0], offsets[1], name))

    expected_offset = 0
    for start, end, name in sorted(spans):
        if start != expected_offset:
            raise ValueError(
                f"tensor data offsets mismatch: {name}; expected start "
                f"{expected_offset}, got {start}"
            )
        expected_offset = end
    if expected_offset != BUNDLE_TENSOR_BYTES:
        raise ValueError(
            "tensor bytes mismatch: "
            f"expected {BUNDLE_TENSOR_BYTES}, got {expected_offset}"
        )
    if payload_bytes is not None and payload_bytes != expected_offset:
        raise ValueError(
            "safetensors payload bytes mismatch: "
            f"expected {expected_offset}, got {payload_bytes}"
        )


def _load_tensors(path: Path, expected: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    loaded: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as archive:
        names = set(archive.keys())
        if names != set(expected):
            missing = sorted(set(expected) - names)
            unexpected = sorted(names - set(expected))
            raise ValueError(
                f"loaded tensor names mismatch: missing {missing}, unexpected {unexpected}"
            )
        for name in sorted(expected):
            tensor = archive.get_tensor(name)
            expected_tensor = expected[name]
            if tensor.dtype != expected_tensor.dtype:
                raise ValueError(
                    f"loaded tensor dtype mismatch: {name}; expected "
                    f"{expected_tensor.dtype}, got {tensor.dtype}"
                )
            if tensor.shape != expected_tensor.shape:
                raise ValueError(
                    f"loaded tensor shape mismatch: {name}; expected "
                    f"{tuple(expected_tensor.shape)}, got {tuple(tensor.shape)}"
                )
            loaded[name] = tensor
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in loaded.values()
    )
    if tensor_bytes != BUNDLE_TENSOR_BYTES:
        raise ValueError(
            f"loaded tensor bytes mismatch: expected {BUNDLE_TENSOR_BYTES}, got {tensor_bytes}"
        )
    return loaded


def _write_private_staging_copy(file_bytes: bytes) -> Path:
    """Create a private, verified copy for safetensors' path-based reader."""
    with tempfile.NamedTemporaryFile(
        mode="w+b", suffix=".safetensors", delete=False
    ) as staging:
        staging.write(file_bytes)
        staging.flush()
        return Path(staging.name)


def load_gamma_ensemble(path: Path, expected_sha256: str) -> GammaEnsemble:
    """Verify and load one exact CPU/eval ten-member gamma ensemble."""
    if not _is_lower_hex_digest(expected_sha256):
        raise GammaEnsembleLoadError(
            "expected_sha256 must be exactly 64 lowercase hexadecimal characters"
        )
    path = Path(path)
    try:
        file_bytes = _read_regular_file_bytes(path, "gamma ensemble asset")
        if path.name != BUNDLE_FILENAME:
            raise ValueError(
                f"gamma ensemble asset filename must be {BUNDLE_FILENAME}: {path}"
            )
        actual_sha256 = _sha256_file(io.BytesIO(file_bytes))
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        expected = GammaEnsemble().eval().state_dict()
        header = read_safetensors_header(file_bytes)
        header_size = int.from_bytes(file_bytes[:8], "little")
        validate_bundle_header(
            header,
            expected,
            payload_bytes=len(file_bytes) - 8 - header_size,
        )
        staging_path = _write_private_staging_copy(file_bytes)
        try:
            tensors = _load_tensors(staging_path, expected)
        finally:
            staging_path.unlink(missing_ok=True)
        model = GammaEnsemble()
        model.load_state_dict(tensors, strict=True)
        return model.eval()
    except GammaEnsembleLoadError:
        raise
    except Exception as error:
        raise GammaEnsembleLoadError(f"unable to load gamma ensemble: {error}") from error
