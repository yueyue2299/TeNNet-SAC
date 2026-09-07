from dataclasses import FrozenInstanceError

import pytest
import torch

from tennetsac.smi_ted_light.load import Smi_ted
from tennetsac.smi_ted_light.tokenizer import MolTranBertTokenizer


@pytest.fixture
def tiny_tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "<bos>\n<eos>\n<pad>\n<mask>\nC\nO\nc\n1\n",
        encoding="utf-8",
    )
    return MolTranBertTokenizer(vocab_file=str(vocab))


@pytest.fixture
def tiny_config():
    return {
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 4,
        "max_len": 12,
        "num_feats": 4,
        "d_dropout": 0.0,
        "n_output": 1,
    }


def test_production_contract_is_exact():
    from tennetsac.smi_ted_light.asset_contract import SMI_TED_LIGHT_CONTRACT

    assert SMI_TED_LIGHT_CONTRACT.architecture == {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "max_len": 202,
        "num_feats": 32,
    }
    assert SMI_TED_LIGHT_CONTRACT.vocab_size == 2393
    assert SMI_TED_LIGHT_CONTRACT.tensor_count == 224
    assert SMI_TED_LIGHT_CONTRACT.tensor_bytes == 656_641_536


def test_production_contract_cannot_be_mutated():
    from tennetsac.smi_ted_light.asset_contract import SMI_TED_LIGHT_CONTRACT

    with pytest.raises(TypeError):
        SMI_TED_LIGHT_CONTRACT.architecture["n_layer"] = 1
    with pytest.raises(TypeError):
        SMI_TED_LIGHT_CONTRACT.prefix_map["encoder.tok_emb."] = "changed."
    with pytest.raises(FrozenInstanceError):
        SMI_TED_LIGHT_CONTRACT.vocab_size = 1


def test_production_config_includes_state_free_dropout():
    from tennetsac.smi_ted_light.inference import production_config

    assert production_config() == {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "max_len": 202,
        "num_feats": 32,
        "d_dropout": 0.2,
    }


def test_inference_model_has_no_reconstruction_modules(tiny_tokenizer, tiny_config):
    from tennetsac.smi_ted_light.inference import SmiTedInferenceModel

    model = SmiTedInferenceModel(tiny_tokenizer, tiny_config)
    assert set(dict(model.named_children())) == {"encoder", "projector"}
    assert not hasattr(model.encoder, "lang_model")
    assert not hasattr(model, "decoder")
    assert not hasattr(model, "net")


def test_projector_obeys_module_device_not_global_cuda(monkeypatch):
    from tennetsac.smi_ted_light.inference import SmiTedProjector

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    projector = SmiTedProjector(feature_size=8, latent_size=4).to("cpu")

    result = projector(torch.ones(2, 8))

    assert result.device.type == "cpu"
    assert not hasattr(projector, "is_cuda_available")


def test_tokenize_moves_tensors_to_model_device(
    tiny_tokenizer, tiny_config, monkeypatch
):
    from tennetsac.smi_ted_light.inference import SmiTedInferenceModel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = SmiTedInferenceModel(tiny_tokenizer, tiny_config).to("cpu")

    idx, mask = model.tokenize(["CCO"])

    assert idx.device.type == "cpu"
    assert mask.device.type == "cpu"


def _copy_embedding_weights(full_model, inference_model):
    from tennetsac.smi_ted_light.asset_contract import SMI_TED_LIGHT_CONTRACT

    mapped_state = {}
    for source_name, tensor in full_model.state_dict().items():
        for source_prefix, target_prefix in SMI_TED_LIGHT_CONTRACT.prefix_map.items():
            if source_name.startswith(source_prefix):
                mapped_state[
                    target_prefix + source_name.removeprefix(source_prefix)
                ] = tensor
                break

    incompatible = inference_model.load_state_dict(mapped_state)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_inference_embeddings_equal_full_model(tiny_tokenizer, tiny_config):
    from tennetsac.smi_ted_light.inference import SmiTedInferenceModel

    torch.manual_seed(7)
    full_model = Smi_ted(tiny_tokenizer, tiny_config)
    full_model.max_len = tiny_config["max_len"]
    full_model.n_embd = tiny_config["n_embd"]
    inference_model = SmiTedInferenceModel(tiny_tokenizer, tiny_config)
    _copy_embedding_weights(full_model, inference_model)
    full_model.eval()
    inference_model.eval()

    full_embeddings = full_model.encode(
        ["CCO", "c1ccccc1"], return_torch=True
    )
    inference_embeddings = inference_model.encode(
        ["CCO", "c1ccccc1"], return_torch=True
    )

    assert torch.equal(full_embeddings, inference_embeddings)
    assert inference_embeddings.shape == (2, tiny_config["n_embd"])


def test_encode_reinserts_nan_for_invalid_smiles(tiny_tokenizer, tiny_config):
    from tennetsac.smi_ted_light.inference import SmiTedInferenceModel

    model = SmiTedInferenceModel(tiny_tokenizer, tiny_config).eval()

    embeddings = model.encode(["invalid"], return_torch=True)

    assert embeddings.shape == (1, tiny_config["n_embd"])
    assert torch.isnan(embeddings).all()


def test_encode_preserves_invalid_smiles_position(tiny_tokenizer, tiny_config):
    from tennetsac.smi_ted_light.inference import SmiTedInferenceModel

    model = SmiTedInferenceModel(tiny_tokenizer, tiny_config).eval()

    embeddings = model.encode(["CCO", "invalid", "C"], return_torch=True)

    assert embeddings.shape == (3, tiny_config["n_embd"])
    assert not torch.isnan(embeddings[0]).any()
    assert torch.isnan(embeddings[1]).all()
    assert not torch.isnan(embeddings[2]).any()
