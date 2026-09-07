import hashlib
import warnings

import pytest

from tennetsac.smi_ted_light import load as smi_ted_load


PARENT_SHA256 = "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375"


class FakeSmiTed:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.loaded_checkpoint = None
        self.is_eval = False

    def _load_checkpoint_data(self, checkpoint):
        self.loaded_checkpoint = checkpoint

    def eval(self):
        self.is_eval = True
        return self


def _asset_entry(parent_sha256="parent-sha256"):
    return {"parent": {"sha256": parent_sha256}}


def test_load_smi_ted_dispatches_safetensors_only_to_inference_loader(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "model.safetensors"
    vocab = tmp_path / "vocab.txt"
    entry = _asset_entry()
    expected_model = object()
    calls = []

    monkeypatch.setattr(
        smi_ted_load,
        "load_smi_ted_inference",
        lambda *args: calls.append(args) or expected_model,
        raising=False,
    )
    monkeypatch.setattr(
        smi_ted_load,
        "load_legacy_smi_ted",
        lambda *args: (_ for _ in ()).throw(
            AssertionError(f"legacy loader called for safetensors: {args}")
        ),
        raising=False,
    )

    model = smi_ted_load.load_smi_ted(checkpoint, vocab, entry)

    assert model is expected_model
    assert calls == [(checkpoint, vocab, entry)]


def test_load_smi_ted_dispatches_pt_only_to_legacy_loader(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    vocab = tmp_path / "vocab.txt"
    entry = _asset_entry("expected-parent-digest")
    expected_model = object()
    calls = []

    monkeypatch.setattr(
        smi_ted_load,
        "load_smi_ted_inference",
        lambda *args: (_ for _ in ()).throw(
            AssertionError(f"inference loader called for legacy checkpoint: {args}")
        ),
        raising=False,
    )

    def fake_legacy_loader(*args):
        calls.append(args)
        warnings.warn(
            "Legacy SMI-TED .pt support will be removed in the next major release; "
            "use the safetensors asset.",
            FutureWarning,
        )
        return expected_model

    monkeypatch.setattr(
        smi_ted_load, "load_legacy_smi_ted", fake_legacy_loader, raising=False
    )

    with pytest.warns(FutureWarning, match="next major"):
        model = smi_ted_load.load_smi_ted(checkpoint, vocab, entry)

    assert model is expected_model
    assert calls == [(checkpoint, vocab, "expected-parent-digest")]


def test_load_smi_ted_rejects_unknown_suffix_without_calling_either_loader(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "model.bin"
    vocab = tmp_path / "vocab.txt"

    def fail_loader(*args):
        raise AssertionError(f"unexpected loader call: {args}")

    monkeypatch.setattr(
        smi_ted_load, "load_smi_ted_inference", fail_loader, raising=False
    )
    monkeypatch.setattr(
        smi_ted_load, "load_legacy_smi_ted", fail_loader, raising=False
    )

    with pytest.raises(
        ValueError, match=r"Unsupported SMI-TED checkpoint format: \.bin"
    ):
        smi_ted_load.load_smi_ted(checkpoint, vocab, _asset_entry())


def test_load_legacy_smi_ted_hashes_before_restricted_deserialization(
    tmp_path, monkeypatch
):
    checkpoint_path = tmp_path / "legacy.pt"
    checkpoint_path.write_bytes(b"exact legacy checkpoint")
    vocab_path = tmp_path / "vocab.txt"
    checkpoint_data = {"hparams": {"max_len": 2}, "MODEL_STATE": {}}
    calls = []

    monkeypatch.setattr(
        smi_ted_load, "MolTranBertTokenizer", lambda vocab_file: ("tokenizer", vocab_file)
    )
    monkeypatch.setattr(smi_ted_load, "Smi_ted", FakeSmiTed)
    monkeypatch.setattr(
        smi_ted_load,
        "_sha256",
        lambda path: calls.append(("sha256", path)) or PARENT_SHA256,
    )

    def fake_torch_load(*args, **kwargs):
        calls.append(("torch.load", args, kwargs))
        return checkpoint_data

    monkeypatch.setattr(smi_ted_load.torch, "load", fake_torch_load)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smi_ted_load.load_legacy_smi_ted(
            checkpoint_path, vocab_path, PARENT_SHA256
        )

    assert calls == [
        ("sha256", checkpoint_path),
        (
            "torch.load",
            (str(checkpoint_path),),
            {"weights_only": True, "mmap": True, "map_location": "cpu"},
        )
    ]
    assert model.loaded_checkpoint is checkpoint_data
    assert model.tokenizer == ("tokenizer", str(vocab_path))
    assert model.is_eval
    assert len(caught) == 1
    assert issubclass(caught[0].category, FutureWarning)
    assert "next major" in str(caught[0].message)
    assert "safetensors" in str(caught[0].message)


def test_load_legacy_smi_ted_rejects_wrong_digest_before_torch_load(
    tmp_path, monkeypatch
):
    checkpoint_path = tmp_path / "legacy.pt"
    checkpoint_path.write_bytes(b"wrong legacy checkpoint")
    actual_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    monkeypatch.setattr(
        smi_ted_load.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("torch.load must not run for a mismatched checkpoint")
        ),
    )

    with pytest.raises(ValueError) as error:
        smi_ted_load.load_legacy_smi_ted(
            checkpoint_path, tmp_path / "vocab.txt", PARENT_SHA256
        )

    assert str(error.value) == (
        f"SHA256 mismatch for {checkpoint_path}: expected {PARENT_SHA256}, "
        f"got {actual_sha256}"
    )


def test_load_legacy_smi_ted_rejects_an_unpinned_expected_digest(
    tmp_path, monkeypatch
):
    checkpoint_path = tmp_path / "legacy.pt"
    checkpoint_path.write_bytes(b"legacy checkpoint")
    monkeypatch.setattr(
        smi_ted_load.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("torch.load must not run for an unpinned digest")
        ),
    )

    with pytest.raises(ValueError, match="exact pinned parent SHA-256"):
        smi_ted_load.load_legacy_smi_ted(
            checkpoint_path, tmp_path / "vocab.txt", "0" * 64
        )


def test_smi_ted_loading_module_has_no_hugging_face_fallback():
    assert "hf_hub_download" not in vars(smi_ted_load)
    assert "huggingface_hub" not in vars(smi_ted_load)
