"""Real local parity checks for the pinned SMI-TED inference asset."""

import gc
import hashlib
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open


PARENT_ENV = "TENNETSAC_SMI_TED_PARENT_CHECKPOINT"
INFERENCE_ENV = "TENNETSAC_SMI_TED_INFERENCE_CHECKPOINT"
VALID_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "ClCCCl", "C[C@H](O)F"]
INVALID_SMILES = ["not-a-smiles"]

pytestmark = pytest.mark.integration


def _required_checkpoint(name: str) -> Path:
    path = Path(os.environ.get(name, ""))
    if not path.is_file():
        pytest.skip(f"{name} does not name a local checkpoint")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_real_smi_ted_inference_asset_matches_the_pinned_parent() -> None:
    parent_path = _required_checkpoint(PARENT_ENV)
    inference_path = _required_checkpoint(INFERENCE_ENV)

    from tennetsac.model_manifest import external_model
    from tennetsac.smi_ted_light.asset_contract import expected_safetensors_metadata
    from tennetsac.smi_ted_light.inference import load_smi_ted_inference
    from tennetsac.smi_ted_light.load import Smi_ted
    from tennetsac.smi_ted_light.tokenizer import MolTranBertTokenizer

    asset_entry = external_model("smi-ted-light")
    assert _sha256(inference_path) == asset_entry["sha256"]
    with safe_open(inference_path, framework="pt", device="cpu") as checkpoint:
        assert checkpoint.metadata() == expected_safetensors_metadata(asset_entry)

    vocab_path = Path(__file__).parents[2] / "src" / "tennetsac" / "smi_ted_light" / "bert_vocab_curated.txt"
    legacy = Smi_ted(MolTranBertTokenizer(str(vocab_path)))
    legacy.load_checkpoint(parent_path)
    legacy.eval()
    legacy_embeddings = legacy.encode(VALID_SMILES, return_torch=True).clone()
    del legacy
    gc.collect()

    inference = load_smi_ted_inference(inference_path, vocab_path, asset_entry).eval()
    inference_embeddings = inference.encode(VALID_SMILES, return_torch=True)

    assert torch.equal(legacy_embeddings, inference_embeddings)
    assert torch.isnan(inference.encode(INVALID_SMILES, return_torch=True)).all()
    assert len(inference.state_dict()) == 224
    assert sum(
        tensor.numel() * tensor.element_size()
        for tensor in inference.state_dict().values()
    ) == 656_641_536
    assert all(
        not name.startswith(("decoder.", "net.", "encoder.lang_model."))
        for name in inference.state_dict()
    )
    assert not hasattr(inference, "decoder")
    assert not hasattr(inference, "net")
    assert not hasattr(inference.encoder, "lang_model")
