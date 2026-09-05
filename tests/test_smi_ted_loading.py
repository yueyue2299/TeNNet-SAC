from pathlib import Path

from smi_ted_light import load as smi_ted_load


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
