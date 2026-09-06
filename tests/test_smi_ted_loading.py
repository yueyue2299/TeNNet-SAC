import hashlib
from pathlib import Path

import pytest

from tennetsac.smi_ted_light import load as smi_ted_load


class FakeSmiTed:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.loaded_checkpoint = None
        self.is_eval = False

    def load_checkpoint(self, checkpoint):
        self.loaded_checkpoint = Path(checkpoint)

    def eval(self):
        self.is_eval = True
        return self


def test_load_smi_ted_prefers_requested_local_checkpoint(tmp_path, monkeypatch):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("<bos>\n<eos>\n<pad>\n<mask>\nC\n", encoding="utf-8")
    checkpoint = tmp_path / "custom.pt"
    checkpoint.touch()

    monkeypatch.setattr(smi_ted_load, "Smi_ted", FakeSmiTed)

    def fail_if_downloaded(**kwargs):
        raise AssertionError(f"unexpected Hugging Face download: {kwargs}")

    monkeypatch.setattr(smi_ted_load, "hf_hub_download", fail_if_downloaded)

    model = smi_ted_load.load_smi_ted(
        folder=tmp_path,
        ckpt_filename=checkpoint.name,
        vocab_filename=vocab.name,
    )

    assert model.loaded_checkpoint == checkpoint
    assert model.is_eval


def test_load_smi_ted_downloads_the_requested_checkpoint_name(tmp_path, monkeypatch):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("<bos>\n<eos>\n<pad>\n<mask>\nC\n", encoding="utf-8")
    downloaded = tmp_path / "downloaded.pt"
    downloaded.touch()
    calls = []

    monkeypatch.setattr(smi_ted_load, "Smi_ted", FakeSmiTed)

    def record_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr(smi_ted_load, "hf_hub_download", record_download)

    model = smi_ted_load.load_smi_ted(
        folder=tmp_path,
        ckpt_filename="custom.pt",
        vocab_filename=vocab.name,
    )

    assert calls == [{"repo_id": "ibm/materials.smi-ted", "filename": "custom.pt"}]
    assert model.loaded_checkpoint == downloaded


def test_load_smi_ted_forwards_pinned_repository_revision(tmp_path, monkeypatch):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("<bos>\n<eos>\n<pad>\n<mask>\nC\n", encoding="utf-8")
    downloaded = tmp_path / "downloaded.pt"
    downloaded.touch()
    calls = []

    monkeypatch.setattr(smi_ted_load, "Smi_ted", FakeSmiTed)
    monkeypatch.setattr(
        smi_ted_load,
        "hf_hub_download",
        lambda **kwargs: calls.append(kwargs) or str(downloaded),
    )

    smi_ted_load.load_smi_ted(
        folder=tmp_path,
        repo_id="ibm/materials.smi-ted",
        revision="414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
        ckpt_filename="smi-ted-Light_40.pt",
        vocab_filename=vocab.name,
    )

    assert calls == [{
        "repo_id": "ibm/materials.smi-ted",
        "filename": "smi-ted-Light_40.pt",
        "revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
    }]


def test_load_smi_ted_rejects_a_local_checkpoint_with_wrong_digest(tmp_path, monkeypatch):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("<bos>\n<eos>\n<pad>\n<mask>\nC\n", encoding="utf-8")
    checkpoint = tmp_path / "smi-ted-Light_40.pt"
    checkpoint.write_bytes(b"incorrect checkpoint")
    expected = "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375"
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    monkeypatch.setattr(
        smi_ted_load.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("torch.load must not run for a mismatched checkpoint")
        ),
    )

    with pytest.raises(ValueError) as error:
        smi_ted_load.load_smi_ted(
            folder=tmp_path,
            ckpt_filename=checkpoint.name,
            vocab_filename=vocab.name,
            expected_sha256=expected,
        )

    assert str(error.value) == (
        f"SHA256 mismatch for smi-ted-Light_40.pt: expected {expected}, got {actual}"
    )


@pytest.mark.parametrize(
    ("revision", "expected_revision"),
    [
        (None, "<not supplied>"),
        ("414c3ea0a8603ef49d1c5bb3db336e09877c01ce", "414c3ea0a8603ef49d1c5bb3db336e09877c01ce"),
    ],
)
def test_load_smi_ted_wraps_download_failure_with_actionable_context(
    tmp_path, monkeypatch, revision, expected_revision
):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("<bos>\n<eos>\n<pad>\n<mask>\nC\n", encoding="utf-8")
    download_error = RuntimeError("offline cache miss")

    monkeypatch.setattr(smi_ted_load, "Smi_ted", FakeSmiTed)
    monkeypatch.setattr(
        smi_ted_load,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(download_error),
    )

    with pytest.raises(RuntimeError) as error:
        smi_ted_load.load_smi_ted(
            folder=tmp_path,
            repo_id="ibm/materials.smi-ted",
            revision=revision,
            ckpt_filename="smi-ted-Light_40.pt",
            vocab_filename=vocab.name,
        )

    message = str(error.value)
    assert str(tmp_path / "smi-ted-Light_40.pt") in message
    assert "ibm/materials.smi-ted" in message
    assert "smi-ted-Light_40.pt" in message
    assert expected_revision in message
    assert "TENNETSAC_SMI_TED_CHECKPOINT" in message
    assert error.value.__cause__ is download_error
