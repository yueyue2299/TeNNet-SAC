import torch

from tennetsac.utils import embedding


class FakeModel:
    def to(self, device):
        self.device = device
        return self

    def encode(self, smiles, return_torch=False):
        assert return_torch is True
        return torch.tensor([[1.0]], device=self.device)


def test_chemberta_embedder_forwards_the_pinned_revision(monkeypatch):
    calls = []

    def tokenizer_from_pretrained(*args, **kwargs):
        calls.append(("tokenizer", args, kwargs))
        return object()

    def model_from_pretrained(*args, **kwargs):
        calls.append(("model", args, kwargs))
        return FakeModel()

    monkeypatch.setattr(
        embedding.RobertaTokenizer, "from_pretrained", tokenizer_from_pretrained
    )
    monkeypatch.setattr(embedding.RobertaModel, "from_pretrained", model_from_pretrained)

    embedding.ChemBERTaEmbedder(
        model_name="DeepChem/ChemBERTa-77M-MLM",
        revision="ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
    )

    assert calls == [
        ("tokenizer", ("DeepChem/ChemBERTa-77M-MLM",), {"revision": "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"}),
        ("model", ("DeepChem/ChemBERTa-77M-MLM",), {"revision": "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"}),
    ]


def test_smi_ted_embedder_forwards_explicit_asset_inputs(monkeypatch, tmp_path):
    checkpoint = tmp_path / "custom.pt"
    vocab = tmp_path / "bert_vocab_curated.txt"
    asset_entry = {
        "name": "smi-ted-light",
        "parent": {"sha256": "parent-digest"},
    }
    calls = []

    def fake_load_smi_ted(**kwargs):
        calls.append(kwargs)
        return FakeModel()

    monkeypatch.setattr(embedding, "load_smi_ted", fake_load_smi_ted)

    embedder = embedding.SMITEDEmbedder(
        checkpoint_path=checkpoint,
        vocab_path=vocab,
        asset_entry=asset_entry,
        device="cpu",
    )

    assert calls == [
        {
            "checkpoint_path": checkpoint,
            "vocab_path": vocab,
            "asset_entry": asset_entry,
        }
    ]
    assert embedder.model.device == "cpu"


def test_smi_ted_embedder_always_returns_cpu_tensor(monkeypatch, tmp_path):
    monkeypatch.setattr(embedding, "load_smi_ted", lambda **kwargs: FakeModel())
    embedder = embedding.SMITEDEmbedder(
        checkpoint_path=tmp_path / "model.safetensors",
        vocab_path=tmp_path / "vocab.txt",
        asset_entry={"parent": {"sha256": "parent-digest"}},
        device="cpu",
    )

    result = embedder(["CCO"])

    assert result.device.type == "cpu"
    assert torch.equal(result, torch.tensor([[1.0]]))
