import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from scripts import _safetensors_canonical as canonical
from scripts import build_smi_ted_inference_asset as converter


def _complete_source_state():
    return {
        "encoder.tok_emb.weight": torch.ones(2, 2),
        "encoder.blocks.layers.0.weight": torch.ones(2, 2),
        "decoder.autoencoder.encoder.fc1.weight": torch.ones(2, 2),
        "encoder.lang_model.head.weight": torch.ones(2, 2),
        "decoder.autoencoder.decoder.rec.weight": torch.ones(2, 2),
        "decoder.lang_model.head.weight": torch.ones(2, 2),
    }


def _parent_checkpoint(source_state=None):
    return {
        "MODEL_STATE": source_state or _complete_source_state(),
        "EPOCHS_RUN": 40,
        "hparams": {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "max_len": 202,
            "num_feats": 32,
        },
    }


def test_select_inference_state_maps_only_required_tensors():
    selected = converter.select_inference_state(_complete_source_state())

    assert set(selected) == {
        "encoder.tok_emb.weight",
        "encoder.blocks.layers.0.weight",
        "projector.fc1.weight",
    }


@pytest.mark.parametrize(
    "unknown_key",
    ["net.weight", "encoder.unknown.weight", "projector.fc1.weight"],
)
def test_select_inference_state_rejects_unknown_source_prefix(unknown_key):
    source = _complete_source_state()
    source[unknown_key] = torch.ones(1)

    with pytest.raises(ValueError, match=f"unknown source key: {unknown_key}"):
        converter.select_inference_state(source)


def test_select_inference_state_rejects_duplicate_normalized_key(monkeypatch):
    monkeypatch.setattr(
        converter,
        "SOURCE_PREFIX_MAP",
        {
            "encoder.tok_emb.": "shared.",
            "encoder.blocks.": "shared.",
            "decoder.autoencoder.encoder.": "projector.",
        },
    )
    source = _complete_source_state()
    source["encoder.tok_emb.weight"] = torch.ones(1)
    source["encoder.blocks.weight"] = torch.ones(1)

    with pytest.raises(ValueError, match="duplicate normalized key: shared.weight"):
        converter.select_inference_state(source)


def test_select_inference_state_rejects_non_tensor_value():
    source = _complete_source_state()
    source["encoder.tok_emb.weight"] = [1.0]

    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        converter.select_inference_state(source)


@pytest.mark.parametrize(
    "missing_prefix",
    [
        "encoder.tok_emb.",
        "encoder.blocks.",
        "decoder.autoencoder.encoder.",
    ],
)
def test_select_inference_state_requires_each_included_prefix(missing_prefix):
    source = {
        name: tensor
        for name, tensor in _complete_source_state().items()
        if not name.startswith(missing_prefix)
    }

    with pytest.raises(ValueError, match=f"missing source prefix: {missing_prefix}"):
        converter.select_inference_state(source)


def test_load_verified_parent_hashes_before_safe_deserialization(
    tmp_path, monkeypatch
):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"controlled checkpoint bytes")
    expected_digest = hashlib.sha256(parent.read_bytes()).hexdigest()
    checkpoint = _parent_checkpoint()
    calls = []

    def fake_torch_load(path, **kwargs):
        calls.append((path, kwargs))
        return checkpoint

    monkeypatch.setattr(converter.torch, "load", fake_torch_load)

    loaded = converter.load_verified_parent(
        parent, expected_sha256=expected_digest
    )

    assert loaded is checkpoint
    assert calls == [
        (
            str(parent),
            {
                "map_location": torch.device("cpu"),
                "weights_only": True,
                "mmap": True,
            },
        )
    ]


def test_load_verified_parent_rejects_digest_before_torch_load(
    tmp_path, monkeypatch
):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"wrong checkpoint")

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("torch.load must not run before digest verification")

    monkeypatch.setattr(converter.torch, "load", fail_if_loaded)

    with pytest.raises(ValueError, match="parent SHA256 mismatch"):
        converter.load_verified_parent(parent, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    "checkpoint",
    [
        {
            "MODEL_STATE": _complete_source_state(),
            "EPOCHS_RUN": 40,
        },
        {
            "EPOCHS_RUN": 40,
            "hparams": _parent_checkpoint()["hparams"],
        },
        {**_parent_checkpoint(), "extra": object()},
    ],
)
def test_validate_parent_checkpoint_requires_exact_top_level_structure(
    checkpoint,
):
    with pytest.raises(ValueError, match="top-level keys"):
        converter.validate_parent_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("name", "wrong_value"),
    [
        ("n_layer", 11),
        ("n_head", 8),
        ("n_embd", 512),
        ("max_len", 128),
        ("num_feats", 16),
    ],
)
def test_validate_parent_checkpoint_requires_exact_architecture(
    name, wrong_value
):
    checkpoint = _parent_checkpoint()
    checkpoint["hparams"][name] = wrong_value

    with pytest.raises(ValueError, match=f"hparams {name} mismatch"):
        converter.validate_parent_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "checkpoint",
    [
        {**_parent_checkpoint(), "hparams": []},
        {**_parent_checkpoint(), "MODEL_STATE": []},
    ],
)
def test_validate_parent_checkpoint_requires_mapping_members(checkpoint):
    with pytest.raises(TypeError, match="must be a mapping"):
        converter.validate_parent_checkpoint(checkpoint)


def test_sha256_file_reads_in_one_mib_chunks():
    payload = b"x" * (converter.HASH_CHUNK_BYTES + 17)

    class RecordingStream:
        def __init__(self):
            self.offset = 0
            self.sizes = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            self.sizes.append(size)
            chunk = payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    stream = RecordingStream()

    class ControlledPath:
        def open(self, mode):
            assert mode == "rb"
            return stream

    assert converter.sha256_file(ControlledPath()) == hashlib.sha256(payload).hexdigest()
    assert stream.sizes == [
        converter.HASH_CHUNK_BYTES,
        converter.HASH_CHUNK_BYTES,
        converter.HASH_CHUNK_BYTES,
    ]


def test_load_verified_parent_reports_restricted_loader_failure(
    tmp_path, monkeypatch
):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"controlled checkpoint bytes")
    expected_digest = hashlib.sha256(parent.read_bytes()).hexdigest()

    def fail_to_load(*args, **kwargs):
        raise RuntimeError("unsupported checkpoint opcode")

    monkeypatch.setattr(converter.torch, "load", fail_to_load)

    with pytest.raises(RuntimeError, match="weights_only=True and mmap=True") as exc:
        converter.load_verified_parent(parent, expected_sha256=expected_digest)

    assert "unsupported checkpoint opcode" in str(exc.value)


def test_build_asset_refuses_preexisting_output_before_parent_access(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")

    def fail_if_parent_accessed(*args, **kwargs):
        raise AssertionError("parent must not be accessed for an existing output")

    monkeypatch.setattr(converter, "load_verified_parent", fail_if_parent_accessed)

    with pytest.raises(FileExistsError, match="output path already exists"):
        converter.build_asset(tmp_path / "parent.pt", output_dir, "abc123")

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert list(output_dir.iterdir()) == [sentinel]


def _controlled_release_state():
    base = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    return {
        "encoder.tok_emb.weight": base[:, ::2],
        "encoder.blocks.layer.weight": torch.ones(2, dtype=torch.float32),
        "projector.fc1.weight": torch.ones(3, dtype=torch.float32),
    }


def _controlled_metadata():
    return {
        "format": "tennetsac-smi-ted-inference",
        "format_version": "1",
        "fixture": "controlled",
    }


def _production_verification_summary():
    return converter.VerificationSummary(
        tensor_count=224,
        tensor_bytes=656_641_536,
        parameter_bytes=656_541_696,
        buffer_bytes=99_840,
    )


def _write_controlled_release(output_dir, parent):
    result = converter.write_release_payload(
        output_dir,
        _controlled_release_state(),
        _controlled_metadata(),
    )
    converter.finalize_release_files(
        output_dir,
        result,
        parent_path=parent,
        converter_commit="abc123def456",
        verification=_production_verification_summary(),
    )
    return result


def test_write_release_files_emits_exact_verified_release_set(tmp_path):
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    parent = tmp_path / "smi-ted-Light_40.pt"

    result = _write_controlled_release(output_dir, parent)

    expected_names = {
        "smi-ted-light-inference-v1.safetensors",
        "IBM-materials-APACHE-2.0.txt",
        "SMI_TED_INFERENCE_PROVENANCE.md",
        "SHA256SUMS",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_names
    assert result.artifact_path == (
        output_dir / "smi-ted-light-inference-v1.safetensors"
    )
    assert result.artifact_size == result.artifact_path.stat().st_size
    assert result.artifact_sha256 == hashlib.sha256(
        result.artifact_path.read_bytes()
    ).hexdigest()
    assert result.tensor_count == 3
    assert result.tensor_bytes == 9 * 4

    written_state = load_file(result.artifact_path, device="cpu")
    assert all(tensor.device.type == "cpu" for tensor in written_state.values())
    assert all(tensor.dtype == torch.float32 for tensor in written_state.values())
    assert all(tensor.is_contiguous() for tensor in written_state.values())
    with safe_open(result.artifact_path, framework="pt", device="cpu") as handle:
        assert handle.metadata() == _controlled_metadata()


@pytest.mark.parametrize(
    "tensor",
    [
        torch.ones(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float16),
        torch.ones(2, dtype=torch.int64),
        torch.ones(2, dtype=torch.bool),
    ],
)
def test_write_release_files_rejects_non_float32_retained_tensor(
    tmp_path, tensor
):
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()

    with pytest.raises(
        TypeError,
        match=(
            "retained tensor encoder.tok_emb.weight must have dtype "
            f"torch.float32, got {tensor.dtype}"
        ),
    ):
        converter.write_release_payload(
            output_dir,
            {"encoder.tok_emb.weight": tensor},
            _controlled_metadata(),
        )

    assert list(output_dir.iterdir()) == []


def test_write_release_files_preserves_exact_float32_values(tmp_path):
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    source_base = torch.tensor(
        [
            [0.0, 91.0, -0.0, 92.0],
            [1.25, 93.0, -3.5, 94.0],
        ],
        dtype=torch.float32,
    )
    source = source_base[:, ::2]
    assert not source.is_contiguous()

    result = converter.write_release_payload(
        output_dir,
        {"encoder.tok_emb.weight": source},
        _controlled_metadata(),
    )

    reloaded = load_file(result.artifact_path, device="cpu")[
        "encoder.tok_emb.weight"
    ]
    assert reloaded.device.type == "cpu"
    assert reloaded.is_contiguous()
    assert reloaded.dtype == torch.float32
    assert torch.equal(
        reloaded.view(torch.int32), source.contiguous().view(torch.int32)
    )


def _raw_safetensors_header(path):
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        return stream.read(header_size)


def test_release_payload_is_reproducible_across_mapping_insertion_orders(
    tmp_path,
):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    alpha = torch.tensor([1.25, -0.0], dtype=torch.float32)
    zeta = torch.tensor([-3.5, 9.0], dtype=torch.float32)
    first_state = {"zeta.weight": zeta, "alpha.weight": alpha}
    second_state = {"alpha.weight": alpha.clone(), "zeta.weight": zeta.clone()}
    first_metadata = {"zeta": "last", "alpha": "first"}
    second_metadata = {"alpha": "first", "zeta": "last"}

    first = converter.write_release_payload(
        first_dir, first_state, first_metadata
    )
    second = converter.write_release_payload(
        second_dir, second_state, second_metadata
    )

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.artifact_path.read_bytes() == second.artifact_path.read_bytes()
    for result in (first, second):
        raw_header = _raw_safetensors_header(result.artifact_path)
        header = json.loads(raw_header)
        assert list(header) == ["__metadata__", "alpha.weight", "zeta.weight"]
        assert list(header["__metadata__"]) == ["alpha", "zeta"]
        assert raw_header.rstrip(b" ") == json.dumps(
            header,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        reloaded = load_file(result.artifact_path, device="cpu")
        assert torch.equal(reloaded["alpha.weight"], alpha)
        assert torch.equal(reloaded["zeta.weight"], zeta)


def test_header_canonicalization_removes_temp_file_when_replace_fails(
    tmp_path, monkeypatch
):
    artifact_path = tmp_path / "payload.safetensors"
    converter.save_file(
        {"zeta.weight": torch.ones(1), "alpha.weight": torch.zeros(1)},
        artifact_path,
        metadata={"zeta": "last", "alpha": "first"},
    )
    original_bytes = artifact_path.read_bytes()

    def reject_replace(temp_path, destination):
        assert destination == artifact_path
        assert Path(temp_path).parent == artifact_path.parent
        raise OSError("replace failed")

    monkeypatch.setattr(canonical.os, "replace", reject_replace)

    with pytest.raises(OSError, match="replace failed"):
        converter._canonicalize_safetensors_header(artifact_path)

    assert artifact_path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [artifact_path]


def test_write_release_files_copies_license_and_hashes_other_three_files(tmp_path):
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    _write_controlled_release(output_dir, tmp_path / "smi-ted-Light_40.pt")

    repository_license = (
        Path(converter.__file__).resolve().parents[1]
        / "licenses"
        / "IBM-materials-APACHE-2.0.txt"
    )
    copied_license = output_dir / "IBM-materials-APACHE-2.0.txt"
    assert copied_license.read_bytes() == repository_license.read_bytes()
    assert hashlib.sha256(copied_license.read_bytes()).hexdigest() == (
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    )

    checksum_lines = (output_dir / "SHA256SUMS").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(checksum_lines) == 3
    expected_filenames = [
        "smi-ted-light-inference-v1.safetensors",
        "IBM-materials-APACHE-2.0.txt",
        "SMI_TED_INFERENCE_PROVENANCE.md",
    ]
    for line, filename in zip(checksum_lines, expected_filenames, strict=True):
        digest, separator, actual_filename = line.partition("  ")
        assert separator == "  "
        assert actual_filename == filename
        assert digest == digest.lower()
        assert len(digest) == 64
        assert digest == hashlib.sha256(
            (output_dir / filename).read_bytes()
        ).hexdigest()
    assert "SHA256SUMS" not in "\n".join(checksum_lines)


def test_write_release_files_records_complete_provenance(tmp_path):
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    parent = tmp_path / "smi-ted-Light_40.pt"
    _write_controlled_release(output_dir, parent)

    provenance = (output_dir / "SMI_TED_INFERENCE_PROVENANCE.md").read_text(
        encoding="utf-8"
    )
    for required_text in (
        "ibm/materials.smi-ted",
        "ibm-research/materials.smi-ted",
        "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
        "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
        "abc123def456",
        "--parent",
        str(parent),
        "--output-dir",
        str(output_dir),
        "--converter-commit",
        "encoder.tok_emb.",
        "encoder.blocks.",
        "decoder.autoencoder.encoder.",
        "encoder.lang_model.",
        "decoder.autoencoder.decoder.",
        "decoder.lang_model.",
        "inference-only derivative",
        "values and float32 dtype were preserved without dtype conversion",
        "n_layer=12",
        "n_head=12",
        "n_embd=768",
        "max_len=202",
        "num_feats=32",
        "Strict production inference-loader reload: PASS",
        "224 tensors",
        "656,641,536 tensor bytes",
        "656,541,696 parameter bytes",
        "99,840 buffer bytes",
    ):
        assert required_text in provenance
    assert "bitwise" not in provenance.lower()
    assert "golden" not in provenance.lower()


def test_build_asset_reloads_before_finalizing_provenance_and_checksums(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "candidate"
    checkpoint = _parent_checkpoint()
    monkeypatch.setattr(
        converter, "load_verified_parent", lambda path: checkpoint
    )
    events = []

    def record_reload(checkpoint_path, vocab_path, asset_entry):
        events.append("reload")
        assert checkpoint_path.is_file()
        assert (output_dir / "IBM-materials-APACHE-2.0.txt").is_file()
        assert not (output_dir / "SMI_TED_INFERENCE_PROVENANCE.md").exists()
        assert not (output_dir / "SHA256SUMS").exists()
        return object()

    monkeypatch.setattr(converter, "load_smi_ted_inference", record_reload)
    monkeypatch.setattr(
        converter,
        "_verified_inference_summary",
        lambda model: _production_verification_summary(),
        raising=False,
    )

    converter.build_asset(tmp_path / "parent.pt", output_dir, "abc123")

    assert events == ["reload"]
    assert (output_dir / "SMI_TED_INFERENCE_PROVENANCE.md").is_file()
    assert (output_dir / "SHA256SUMS").is_file()


def test_reload_failure_occurs_before_provenance_and_checksum_finalization(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "candidate"
    checkpoint = _parent_checkpoint()
    monkeypatch.setattr(
        converter, "load_verified_parent", lambda path: checkpoint
    )

    def reject_reload(checkpoint_path, vocab_path, asset_entry):
        assert checkpoint_path.is_file()
        assert (output_dir / "IBM-materials-APACHE-2.0.txt").is_file()
        assert not (output_dir / "SMI_TED_INFERENCE_PROVENANCE.md").exists()
        assert not (output_dir / "SHA256SUMS").exists()
        raise ValueError("reload rejected")

    monkeypatch.setattr(converter, "load_smi_ted_inference", reject_reload)

    with pytest.raises(ValueError, match="reload rejected"):
        converter.build_asset(tmp_path / "parent.pt", output_dir, "abc123")

    assert not output_dir.exists()


def test_build_asset_reloads_written_asset_and_returns_frozen_result(
    tmp_path, monkeypatch
):
    parent = tmp_path / "parent.pt"
    output_dir = tmp_path / "candidate"
    checkpoint = _parent_checkpoint()
    calls = []
    monkeypatch.setattr(
        converter, "load_verified_parent", lambda path: checkpoint
    )

    def record_reload(checkpoint_path, vocab_path, asset_entry):
        calls.append((checkpoint_path, vocab_path, asset_entry))
        return object()

    monkeypatch.setattr(converter, "load_smi_ted_inference", record_reload)
    monkeypatch.setattr(
        converter,
        "_verified_inference_summary",
        lambda model: _production_verification_summary(),
    )

    result = converter.build_asset(parent, output_dir, "abc123def456")

    assert len(calls) == 1
    assert calls[0][0] == result.artifact_path
    assert calls[0][1].name == "bert_vocab_curated.txt"
    assert calls[0][2] == {
        "architecture": {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "max_len": 202,
            "num_feats": 32,
        },
        "vocab_size": 2393,
        "state_tensor_count": 3,
        "state_tensor_bytes": 3 * 2 * 2 * 4,
    }
    with pytest.raises(FrozenInstanceError):
        result.tensor_count = 4


@pytest.mark.parametrize("failure_stage", ["write", "reload"])
def test_build_asset_removes_new_output_after_any_failure(
    tmp_path, monkeypatch, failure_stage
):
    output_dir = tmp_path / "candidate"
    checkpoint = _parent_checkpoint()
    monkeypatch.setattr(
        converter, "load_verified_parent", lambda path: checkpoint
    )
    if failure_stage == "write":
        monkeypatch.setattr(
            converter,
            "save_file",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
        )
    else:
        monkeypatch.setattr(
            converter,
            "load_smi_ted_inference",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("reload rejected")
            ),
        )

    with pytest.raises((OSError, ValueError)):
        converter.build_asset(tmp_path / "parent.pt", output_dir, "abc123")

    assert not output_dir.exists()


def test_build_asset_removes_output_after_header_canonicalization_failure(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "candidate"
    checkpoint = _parent_checkpoint()
    monkeypatch.setattr(
        converter, "load_verified_parent", lambda path: checkpoint
    )
    monkeypatch.setattr(
        converter,
        "_canonicalize_safetensors_header",
        lambda path: (_ for _ in ()).throw(OSError("replace failed")),
        raising=False,
    )
    monkeypatch.setattr(
        converter, "load_smi_ted_inference", lambda *args: object()
    )
    monkeypatch.setattr(
        converter,
        "_verified_inference_summary",
        lambda model: _production_verification_summary(),
    )

    with pytest.raises(OSError, match="replace failed"):
        converter.build_asset(tmp_path / "parent.pt", output_dir, "abc123")

    assert not output_dir.exists()


def test_cli_prints_stable_build_result_fields(tmp_path, monkeypatch, capsys):
    artifact = tmp_path / "candidate" / "asset.safetensors"
    result = converter.BuildResult(
        artifact_path=artifact,
        artifact_size=123,
        artifact_sha256="a" * 64,
        tensor_count=3,
        tensor_bytes=36,
    )
    monkeypatch.setattr(converter, "build_asset", lambda *args: result)

    assert converter.main(
        [
            "--parent",
            str(tmp_path / "parent.pt"),
            "--output-dir",
            str(tmp_path / "candidate"),
            "--converter-commit",
            "abc123",
        ]
    ) == 0

    assert capsys.readouterr().out.splitlines() == [
        f"artifact_path={artifact}",
        "artifact_size=123",
        f"artifact_sha256={'a' * 64}",
        "tensor_count=3",
        "tensor_bytes=36",
    ]
