import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import tennetsac.gamma_ensemble as loader
from scripts._safetensors_canonical import _canonicalize_safetensors_header
from tennetsac.gamma_ensemble import (
    BUNDLE_FILENAME,
    BUNDLE_METADATA,
    GammaEnsembleLoadError,
    load_gamma_ensemble,
)
from tennetsac.models.GammaEnsemble import GammaEnsemble


@pytest.fixture
def valid_bundle(tmp_path):
    """A canonical bundle whose identity is verified independently by sha256."""
    path = tmp_path / BUNDLE_FILENAME
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in GammaEnsemble().eval().state_dict().items()
    }
    save_file(state, path, metadata=BUNDLE_METADATA)
    _canonicalize_safetensors_header(path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_header(path: Path, header: bytes) -> None:
    data = path.read_bytes()
    original_size = int.from_bytes(data[:8], "little")
    rewritten_size = max(original_size, len(header))
    path.write_bytes(
        rewritten_size.to_bytes(8, "little")
        + header.ljust(rewritten_size, b" ")
        + data[8 + original_size :]
    )


def _rehash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _header(path: Path) -> dict:
    data = path.read_bytes()
    header_size = int.from_bytes(data[:8], "little")
    return json.loads(data[8 : 8 + header_size])


def test_loader_returns_exact_eval_ensemble(valid_bundle):
    bundle, digest = valid_bundle

    model = load_gamma_ensemble(bundle, digest)

    assert isinstance(model, GammaEnsemble)
    assert model.training is False
    assert len(model.heads) == 10
    assert len(model.state_dict()) == 93


def test_loader_rejects_digest_mismatch_before_safetensors_open(
    monkeypatch, valid_bundle
):
    bundle, _ = valid_bundle
    opens = []
    monkeypatch.setattr(loader, "safe_open", lambda *args, **kwargs: opens.append(args))

    with pytest.raises(GammaEnsembleLoadError, match="sha256 mismatch"):
        load_gamma_ensemble(bundle, "0" * 64)

    assert opens == []


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, 0, None])
def test_loader_rejects_a_noncanonical_expected_digest(valid_bundle, digest):
    bundle, _ = valid_bundle

    with pytest.raises(GammaEnsembleLoadError, match="expected_sha256"):
        load_gamma_ensemble(bundle, digest)


def test_loader_rejects_symlink_and_non_regular_paths(tmp_path, valid_bundle):
    bundle, digest = valid_bundle
    symlink = tmp_path / "linked.safetensors"
    symlink.symlink_to(bundle)
    directory = tmp_path / "directory.safetensors"
    directory.mkdir()

    with pytest.raises(GammaEnsembleLoadError, match="symlink"):
        load_gamma_ensemble(symlink, digest)
    with pytest.raises(GammaEnsembleLoadError, match="regular file"):
        load_gamma_ensemble(directory, digest)


def test_loader_rejects_duplicate_header_tensor_names_before_safetensors_open(
    monkeypatch, valid_bundle
):
    bundle, _ = valid_bundle
    raw = bundle.read_bytes()
    header_size = int.from_bytes(raw[:8], "little")
    header = raw[8 : 8 + header_size].rstrip()
    duplicate = header[:-1] + b',"trunk.bn2.bias":{"dtype":"F32","shape":[256],"data_offsets":[0,1024]}}'
    _rewrite_header(bundle, duplicate)

    with pytest.raises(GammaEnsembleLoadError, match="duplicate.*trunk.bn2.bias") as caught:
        load_gamma_ensemble(bundle, _rehash(bundle))

    assert isinstance(caught.value.__cause__, ValueError)


def test_loader_rejects_malformed_header_and_preserves_json_cause(valid_bundle):
    bundle, _ = valid_bundle
    _rewrite_header(bundle, b"{")

    with pytest.raises(GammaEnsembleLoadError, match="invalid safetensors header") as caught:
        load_gamma_ensemble(bundle, _rehash(bundle))

    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


def test_loader_rejects_wrong_metadata(valid_bundle):
    bundle, _ = valid_bundle
    header = _header(bundle)
    header["__metadata__"]["member_count"] = "9"
    _rewrite_header(bundle, json.dumps(header, separators=(",", ":")).encode())

    with pytest.raises(GammaEnsembleLoadError, match="metadata mismatch"):
        load_gamma_ensemble(bundle, _rehash(bundle))


@pytest.mark.parametrize("corruption", ["missing", "extra"])
def test_loader_rejects_missing_or_extra_tensor_name(valid_bundle, corruption):
    bundle, _ = valid_bundle
    header = _header(bundle)
    if corruption == "missing":
        del header["heads.0.0.bias"]
    else:
        header["unexpected"] = header["heads.0.0.bias"]
    _rewrite_header(bundle, json.dumps(header, separators=(",", ":")).encode())

    with pytest.raises(GammaEnsembleLoadError, match="tensor names mismatch"):
        load_gamma_ensemble(bundle, _rehash(bundle))


@pytest.mark.parametrize("corruption", ["shape", "dtype"])
def test_loader_rejects_wrong_tensor_shape_or_dtype(valid_bundle, corruption):
    bundle, _ = valid_bundle
    header = _header(bundle)
    descriptor = header["heads.0.0.bias"]
    descriptor["shape"] = [255] if corruption == "shape" else descriptor["shape"]
    descriptor["dtype"] = "F64" if corruption == "dtype" else descriptor["dtype"]
    _rewrite_header(bundle, json.dumps(header, separators=(",", ":")).encode())

    with pytest.raises(GammaEnsembleLoadError, match=f"tensor {corruption} mismatch"):
        load_gamma_ensemble(bundle, _rehash(bundle))


def test_loader_rejects_wrong_tensor_count(valid_bundle):
    bundle, _ = valid_bundle
    header = _header(bundle)
    del header["heads.0.0.bias"]
    _rewrite_header(bundle, json.dumps(header, separators=(",", ":")).encode())

    with pytest.raises(GammaEnsembleLoadError, match="tensor names mismatch"):
        load_gamma_ensemble(bundle, _rehash(bundle))


def test_loader_rejects_wrong_tensor_bytes(valid_bundle):
    bundle, _ = valid_bundle
    header = _header(bundle)
    header["heads.0.0.bias"]["data_offsets"][1] -= 4
    _rewrite_header(bundle, json.dumps(header, separators=(",", ":")).encode())
    with pytest.raises(GammaEnsembleLoadError, match="tensor bytes mismatch"):
        load_gamma_ensemble(bundle, _rehash(bundle))


def test_loader_preserves_safetensors_open_cause(monkeypatch, valid_bundle):
    bundle, digest = valid_bundle

    def fail_open(*args, **kwargs):
        raise OSError("synthetic safe_open failure")

    monkeypatch.setattr(loader, "safe_open", fail_open)
    with pytest.raises(GammaEnsembleLoadError, match="unable to load") as caught:
        load_gamma_ensemble(bundle, digest)

    assert isinstance(caught.value.__cause__, OSError)
