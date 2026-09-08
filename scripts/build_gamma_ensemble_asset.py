#!/usr/bin/env python3
"""Build the pinned ten-member gamma ensemble without networking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from tennetsac.gamma_ensemble import BUNDLE_METADATA, BUNDLE_TENSOR_BYTES
from tennetsac.models.Prf2Gamma import Prf_to_Seg_Model

if __package__:
    from scripts._safetensors_canonical import _canonicalize_safetensors_header
else:
    from _safetensors_canonical import _canonicalize_safetensors_header


HASH_CHUNK_BYTES = 1024 * 1024
MEMBER_COUNT = 10
FINAL_PREFIX = "model_final."
PRODUCTION_TENSOR_BYTES = BUNDLE_TENSOR_BYTES
CONTRACT_KEYS = {
    "format_version",
    "source_bundle_version",
    "source_commit",
    "members",
}
MEMBER_KEYS = {"member", "path", "sha256"}


@dataclass(frozen=True)
class BundleReport:
    path: Path
    sha256: str
    tensor_count: int
    tensor_bytes: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(
    value: Any,
    expected: set[str],
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{description} keys mismatch: "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{description} keys must be strings")
    return value


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_source_contract(path: Path) -> Mapping[str, Any]:
    if path.is_symlink():
        raise ValueError(f"source contract must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"source contract is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid source contract JSON: {path}") from error
    contract = _require_exact_keys(payload, CONTRACT_KEYS, "source contract")
    if type(contract["format_version"]) is not int or contract["format_version"] != 1:
        raise ValueError("source contract format_version must be 1")
    if contract["source_bundle_version"] != "1.0.0":
        raise ValueError("source contract source_bundle_version must be '1.0.0'")
    if not _is_lower_hex(contract["source_commit"], 40):
        raise ValueError("source contract source_commit must be 40 lowercase hex characters")

    members = contract["members"]
    if not isinstance(members, list):
        raise TypeError("source contract members must be a JSON array")
    if len(members) != MEMBER_COUNT:
        raise ValueError(f"source contract must contain exactly {MEMBER_COUNT} members")
    for expected_number, raw_entry in enumerate(members, start=1):
        entry = _require_exact_keys(
            raw_entry,
            MEMBER_KEYS,
            f"source contract member {expected_number}",
        )
        if type(entry["member"]) is not int or entry["member"] != expected_number:
            raise ValueError(
                "source contract members must be ordered exactly as "
                f"{list(range(1, MEMBER_COUNT + 1))}"
            )
        expected_path = f"ckpt_files/fine-tuned/{expected_number}.ckpt"
        if entry["path"] != expected_path:
            raise ValueError(
                f"source contract member {expected_number} path mismatch: "
                f"expected {expected_path}, got {entry['path']}"
            )
        if not _is_lower_hex(entry["sha256"], 64):
            raise ValueError(
                f"source contract member {expected_number} sha256 must be "
                "64 lowercase hex characters"
            )
    return contract


def _expected_legacy_state() -> Mapping[str, torch.Tensor]:
    return Prf_to_Seg_Model().eval().state_dict()


def _validate_source_state(
    state: Any,
    expected: Mapping[str, torch.Tensor],
    logical_path: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint state must be a mapping: {logical_path}")
    actual_keys = set(state)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys, key=str)
        raise ValueError(
            f"state keys mismatch for {logical_path}: "
            f"missing {missing}, unexpected {unexpected}"
        )
    for key in sorted(expected):
        tensor = state[key]
        expected_tensor = expected[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state value must be a tensor for {logical_path}: {key}")
        if tensor.dtype != expected_tensor.dtype:
            raise ValueError(
                f"tensor dtype mismatch for {logical_path}: {key}; "
                f"expected {expected_tensor.dtype}, got {tensor.dtype}"
            )
        if tensor.shape != expected_tensor.shape:
            raise ValueError(
                f"tensor shape mismatch for {logical_path}: {key}; "
                f"expected {tuple(expected_tensor.shape)}, got {tuple(tensor.shape)}"
            )
    return state


def _tensors_have_identical_bytes(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    """Return whether two tensors have the same dtype, shape, and bytes."""
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            right.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def _load_verified_members(
    source_dir: Path,
    contract: Mapping[str, Any],
) -> list[Mapping[str, torch.Tensor]]:
    if source_dir.is_symlink():
        raise ValueError(f"source directory must not be a symlink: {source_dir}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")

    expected_schema = _expected_legacy_state()
    source_paths: list[tuple[Mapping[str, Any], Path]] = []
    for entry in contract["members"]:
        logical_path = entry["path"]
        source_path = source_dir / Path(logical_path).name
        if source_path.is_symlink():
            raise ValueError(f"source must not be a symlink: {logical_path}")
        if not source_path.is_file():
            raise FileNotFoundError(f"source checkpoint is not a file: {logical_path}")
        source_paths.append((entry, source_path))

    expected_names = {f"{number}.ckpt" for number in range(1, MEMBER_COUNT + 1)}
    actual_names = {path.name for path in source_dir.iterdir()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            "unexpected checkpoint files in source directory: "
            f"missing {missing}, unexpected {unexpected}"
        )

    members = []
    for entry, source_path in source_paths:
        logical_path = entry["path"]
        actual_sha256 = _sha256_file(source_path)
        if actual_sha256 != entry["sha256"]:
            raise ValueError(
                f"SHA256 mismatch for {logical_path}: "
                f"expected {entry['sha256']}, got {actual_sha256}"
            )
        state = torch.load(
            str(source_path),
            map_location="cpu",
            weights_only=True,
        )
        members.append(_validate_source_state(state, expected_schema, logical_path))
    return members


def _build_bundle_state(
    members: list[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    first = members[0]
    shared_keys = sorted(key for key in first if not key.startswith(FINAL_PREFIX))
    head_keys = sorted(key for key in first if key.startswith(FINAL_PREFIX))
    if len(shared_keys) != 33 or len(head_keys) != 6:
        raise ValueError(
            "legacy gamma architecture key counts mismatch: "
            f"expected 33 shared and 6 head, got {len(shared_keys)} and {len(head_keys)}"
        )

    for member_number, member in enumerate(members[1:], start=2):
        for key in shared_keys:
            if not _tensors_have_identical_bytes(member[key], first[key]):
                raise ValueError(
                    f"shared tensor differs for member {member_number}: {key}"
                )

    mapped: dict[str, torch.Tensor] = {}
    for key in shared_keys:
        mapped[f"trunk.{key}"] = first[key].detach().cpu().contiguous()
    for member_index, member in enumerate(members):
        for key in head_keys:
            suffix = key.removeprefix(FINAL_PREFIX)
            mapped[f"heads.{member_index}.{suffix}"] = (
                member[key].detach().cpu().contiguous()
            )
    return {key: mapped[key] for key in sorted(mapped)}


def _verify_saved_bundle(
    path: Path,
    expected: Mapping[str, torch.Tensor],
) -> None:
    actual = load_file(path, device="cpu")
    if set(actual) != set(expected):
        raise ValueError(
            "post-save state keys mismatch: "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )
    for key in sorted(expected):
        expected_tensor = expected[key]
        actual_tensor = actual[key]
        if actual_tensor.dtype != expected_tensor.dtype:
            raise ValueError(f"post-save tensor dtype differs: {key}")
        if actual_tensor.shape != expected_tensor.shape:
            raise ValueError(f"post-save tensor shape differs: {key}")
        if not _tensors_have_identical_bytes(actual_tensor, expected_tensor):
            raise ValueError(f"post-save tensor differs: {key}")


def build_bundle(
    source_dir: Path,
    output_path: Path,
    source_contract_path: Path,
) -> BundleReport:
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    source_contract_path = Path(source_contract_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"output path already exists: {output_path}")

    contract = _load_source_contract(source_contract_path)
    members = _load_verified_members(source_dir, contract)
    bundle_state = _build_bundle_state(members)
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in bundle_state.values()
    )
    if tensor_bytes != PRODUCTION_TENSOR_BYTES:
        raise ValueError(
            "production gamma ensemble tensor bytes mismatch: "
            f"expected {PRODUCTION_TENSOR_BYTES}, got {tensor_bytes}"
        )
    metadata = BUNDLE_METADATA

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        save_file(bundle_state, temporary_path, metadata=metadata)
        _canonicalize_safetensors_header(temporary_path)
        _verify_saved_bundle(temporary_path, bundle_state)
        sha256 = _sha256_file(temporary_path)
        os.replace(temporary_path, output_path)
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return BundleReport(
        path=output_path,
        sha256=sha256,
        tensor_count=len(bundle_state),
        tensor_bytes=tensor_bytes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the deterministic ten-member gamma ensemble asset."
    )
    parser.add_argument("source_dir", type=Path, metavar="SOURCE_DIR")
    parser.add_argument("output_path", type=Path, metavar="OUTPUT_PATH")
    parser.add_argument("--source-contract", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_bundle(args.source_dir, args.output_path, args.source_contract)
    print(f"path={report.path}")
    print(f"sha256={report.sha256}")
    print(f"tensor_count={report.tensor_count}")
    print(f"tensor_bytes={report.tensor_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
