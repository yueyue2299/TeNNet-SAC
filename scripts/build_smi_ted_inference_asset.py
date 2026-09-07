#!/usr/bin/env python3
"""Build the pinned SMI-TED inference-only release asset without networking."""

from __future__ import annotations

import argparse
import hashlib
import shlex
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from tennetsac.smi_ted_light.asset_contract import (
    SMI_TED_LIGHT_CONTRACT,
    expected_safetensors_metadata,
)
from tennetsac.smi_ted_light.inference import load_smi_ted_inference


HASH_CHUNK_BYTES = 1024 * 1024
SOURCE_PREFIX_MAP = SMI_TED_LIGHT_CONTRACT.prefix_map
ALLOWED_EXCLUDED_PREFIXES = SMI_TED_LIGHT_CONTRACT.allowed_excluded_prefixes
ARTIFACT_FILENAME = "smi-ted-light-inference-v1.safetensors"
LICENSE_FILENAME = "IBM-materials-APACHE-2.0.txt"
PROVENANCE_FILENAME = "SMI_TED_INFERENCE_PROVENANCE.md"
CHECKSUM_FILENAME = "SHA256SUMS"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BuildResult:
    artifact_path: Path
    artifact_size: int
    artifact_sha256: str
    tensor_count: int
    tensor_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parent_checkpoint(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("parent checkpoint must be a mapping")
    expected_keys = {"MODEL_STATE", "EPOCHS_RUN", "hparams"}
    actual_keys = set(checkpoint)
    if actual_keys != expected_keys:
        raise ValueError(
            "parent checkpoint top-level keys mismatch: "
            f"expected {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )

    hparams = checkpoint["hparams"]
    if not isinstance(hparams, Mapping):
        raise TypeError("parent checkpoint hparams must be a mapping")
    for name, expected_value in SMI_TED_LIGHT_CONTRACT.architecture.items():
        actual_value = hparams.get(name)
        if actual_value != expected_value:
            raise ValueError(
                f"parent checkpoint hparams {name} mismatch: "
                f"expected {expected_value}, got {actual_value}"
            )

    state_dict = checkpoint["MODEL_STATE"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("parent checkpoint MODEL_STATE must be a mapping")
    return state_dict


def load_verified_parent(
    parent_path: Path | str,
    *,
    expected_sha256: str = SMI_TED_LIGHT_CONTRACT.parent_sha256,
) -> Mapping[str, Any]:
    parent_path = Path(parent_path)
    actual_sha256 = sha256_file(parent_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"parent SHA256 mismatch for {parent_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    try:
        checkpoint = torch.load(
            str(parent_path),
            map_location=torch.device("cpu"),
            weights_only=True,
            mmap=True,
        )
    except Exception as error:
        raise RuntimeError(
            "Unable to deserialize the verified SMI-TED parent with "
            f"PyTorch {torch.__version__} using weights_only=True and mmap=True: "
            f"{error}"
        ) from error
    validate_parent_checkpoint(checkpoint)
    return checkpoint


def select_inference_state(
    source_state: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if not isinstance(source_state, Mapping):
        raise TypeError("source state must be a mapping")

    selected: dict[str, torch.Tensor] = {}
    seen_prefixes: set[str] = set()
    for source_name, tensor in source_state.items():
        if not isinstance(source_name, str):
            raise TypeError("source state keys must be strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"source state value for {source_name} must be a torch.Tensor")

        matched_prefix = next(
            (
                source_prefix
                for source_prefix in SOURCE_PREFIX_MAP
                if source_name.startswith(source_prefix)
            ),
            None,
        )
        if matched_prefix is not None:
            target_prefix = SOURCE_PREFIX_MAP[matched_prefix]
            normalized_name = target_prefix + source_name[len(matched_prefix) :]
            if normalized_name in selected:
                raise ValueError(f"duplicate normalized key: {normalized_name}")
            selected[normalized_name] = tensor
            seen_prefixes.add(matched_prefix)
            continue


        if source_name.startswith(ALLOWED_EXCLUDED_PREFIXES):
            continue
        raise ValueError(f"unknown source key: {source_name}")

    for source_prefix in SOURCE_PREFIX_MAP:
        if source_prefix not in seen_prefixes:
            raise ValueError(f"missing source prefix: {source_prefix}")
    return selected


def _prepare_release_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    prepared = {}
    for name, tensor in state.items():
        if tensor.dtype != torch.float32:
            raise TypeError(
                f"retained tensor {name} must have dtype torch.float32, "
                f"got {tensor.dtype}"
            )
        prepared[name] = tensor.detach().cpu().contiguous()
    return prepared


def _provenance_text(
    *,
    parent_path: Path,
    output_dir: Path,
    converter_commit: str,
    artifact_sha256: str,
) -> str:
    command = " ".join(
        shlex.quote(value)
        for value in (
            "python",
            "scripts/build_smi_ted_inference_asset.py",
            "--parent",
            str(parent_path),
            "--output-dir",
            str(output_dir),
            "--converter-commit",
            converter_commit,
        )
    )
    included = "\n".join(
        f"- `{prefix}` -> `{SOURCE_PREFIX_MAP[prefix]}`"
        for prefix in SOURCE_PREFIX_MAP
    )
    excluded = "\n".join(
        f"- `{prefix}`" for prefix in ALLOWED_EXCLUDED_PREFIXES
    )
    contract = SMI_TED_LIGHT_CONTRACT
    return f"""# SMI-TED Inference Asset Provenance

This file documents a TeNNet-SAC inference-only derivative of the pinned
SMI-TED Light checkpoint. It is not an IBM-published checkpoint and does not
imply IBM endorsement.

## Parent checkpoint

- Historical repository ID: `{contract.parent_repository_historical}`
- Canonical repository ID: `{contract.parent_repository_canonical}`
- Revision: `{contract.parent_revision}`
- Filename: `{contract.parent_filename}`
- SHA-256: `{contract.parent_sha256}`

## Conversion

- Converter commit: `{converter_commit}`
- Pruning rule version: `1`
- Derived artifact: `{ARTIFACT_FILENAME}`
- Derived artifact SHA-256: `{artifact_sha256}`

Reproduction command:

```console
{command}
```

Included source prefixes and key mappings:

{included}

Recognized excluded source prefixes:

{excluded}

All retained tensor values and float32 dtype were preserved without dtype conversion.
Tensors were detached, moved to CPU, made contiguous, and serialized with
safetensors. The reconstruction decoder, language-model heads, and other
unrecognized state are not included.
"""


def write_release_files(
    output_dir: Path | str,
    state: Mapping[str, torch.Tensor],
    metadata: Mapping[str, str],
    *,
    parent_path: Path | str,
    converter_commit: str,
) -> BuildResult:
    output_dir = Path(output_dir)
    parent_path = Path(parent_path)
    release_state = _prepare_release_state(state)
    artifact_path = output_dir / ARTIFACT_FILENAME
    save_file(release_state, artifact_path, metadata=dict(metadata))
    artifact_sha256 = sha256_file(artifact_path)

    license_path = output_dir / LICENSE_FILENAME
    shutil.copyfile(
        REPOSITORY_ROOT / "licenses" / LICENSE_FILENAME,
        license_path,
    )

    provenance_path = output_dir / PROVENANCE_FILENAME
    provenance_path.write_text(
        _provenance_text(
            parent_path=parent_path,
            output_dir=output_dir,
            converter_commit=converter_commit,
            artifact_sha256=artifact_sha256,
        ),
        encoding="utf-8",
    )

    checksum_paths = (artifact_path, license_path, provenance_path)
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths
    )
    (output_dir / CHECKSUM_FILENAME).write_text(checksum_text, encoding="utf-8")

    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in release_state.values()
    )
    return BuildResult(
        artifact_path=artifact_path,
        artifact_size=artifact_path.stat().st_size,
        artifact_sha256=artifact_sha256,
        tensor_count=len(release_state),
        tensor_bytes=tensor_bytes,
    )


def build_asset(
    parent_path: Path | str,
    output_dir: Path | str,
    converter_commit: str,
) -> BuildResult:
    parent_path = Path(parent_path)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output path already exists: {output_dir}")

    checkpoint = load_verified_parent(parent_path)
    source_state = validate_parent_checkpoint(checkpoint)
    selected_state = select_inference_state(source_state)
    release_state = _prepare_release_state(selected_state)
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in release_state.values()
    )
    asset_entry = {
        "architecture": dict(SMI_TED_LIGHT_CONTRACT.architecture),
        "vocab_size": SMI_TED_LIGHT_CONTRACT.vocab_size,
        "state_tensor_count": len(release_state),
        "state_tensor_bytes": tensor_bytes,
    }
    metadata = expected_safetensors_metadata(asset_entry)

    output_dir.mkdir()
    try:
        result = write_release_files(
            output_dir,
            release_state,
            metadata,
            parent_path=parent_path,
            converter_commit=converter_commit,
        )
        vocab_path = (
            REPOSITORY_ROOT
            / "src"
            / "tennetsac"
            / "smi_ted_light"
            / "bert_vocab_curated.txt"
        )
        load_smi_ted_inference(result.artifact_path, vocab_path, asset_entry)
        return result
    except BaseException:
        shutil.rmtree(output_dir)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the pinned SMI-TED inference-only release asset."
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--converter-commit", required=True)
    args = parser.parse_args(argv)

    result = build_asset(args.parent, args.output_dir, args.converter_commit)
    print(f"artifact_path={result.artifact_path}")
    print(f"artifact_size={result.artifact_size}")
    print(f"artifact_sha256={result.artifact_sha256}")
    print(f"tensor_count={result.tensor_count}")
    print(f"tensor_bytes={result.tensor_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
