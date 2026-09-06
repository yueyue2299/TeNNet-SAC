from tennetsac.utils import embedding


class FakeModel:
    def to(self, device):
        self.device = device
        return self


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


def test_smi_ted_embedder_forwards_pinned_checkpoint_provenance(monkeypatch, tmp_path):
    calls = []

    def fake_load_smi_ted(**kwargs):
        calls.append(kwargs)
        return FakeModel()

    monkeypatch.setattr(embedding, "load_smi_ted", fake_load_smi_ted)

    embedding.SMITEDEmbedder(
        model_dir=tmp_path,
        repo_id="ibm/materials.smi-ted",
        revision="414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
        ckpt_name="smi-ted-Light_40.pt",
        expected_sha256="baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
    )

    assert calls == [{
        "folder": tmp_path,
        "repo_id": "ibm/materials.smi-ted",
        "revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
        "ckpt_filename": "smi-ted-Light_40.pt",
        "expected_sha256": "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
    }]
