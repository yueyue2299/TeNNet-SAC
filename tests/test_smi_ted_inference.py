from dataclasses import FrozenInstanceError

import pytest
import torch
from safetensors.torch import save_file

from tennetsac.smi_ted_light.load import Smi_ted
from tennetsac.smi_ted_light.tokenizer import MolTranBertTokenizer


@pytest.fixture
def tiny_vocab_path(tmp_path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(
        "<bos>\n<eos>\n<pad>\n<mask>\nC\nO\nc\n1\n",
        encoding="utf-8",
    )
    return vocab_path


@pytest.fixture
def tiny_tokenizer(tiny_vocab_path):
    return MolTranBertTokenizer(vocab_file=str(tiny_vocab_path))


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


@pytest.fixture
def tiny_model_factory(tiny_tokenizer, tiny_config):
    from tennetsac.smi_ted_light.inference import SmiTedInferenceModel

    return lambda: SmiTedInferenceModel(tiny_tokenizer, tiny_config)


@pytest.fixture
def tiny_model_entry(tiny_model_factory, tiny_config):
    state = tiny_model_factory().state_dict()
    return {
        "architecture": {
            name: tiny_config[name]
            for name in ("n_layer", "n_head", "n_embd", "max_len", "num_feats")
        },
        "vocab_size": 8,
        "state_tensor_count": len(state),
        "state_tensor_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in state.values()
        ),
    }


def test_safetensors_metadata_is_exact_and_has_no_self_digest(tiny_model_entry):
    from tennetsac.smi_ted_light.asset_contract import (
        expected_safetensors_metadata,
    )

    assert expected_safetensors_metadata(tiny_model_entry) == {
        "format": "tennetsac-smi-ted-inference",
        "format_version": "1",
        "upstream_repository": "ibm-research/materials.smi-ted",
        "upstream_revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
        "upstream_filename": "smi-ted-Light_40.pt",
        "upstream_sha256": (
            "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375"
        ),
        "pruning_rule_version": "1",
        "included_prefixes": (
            '["encoder.tok_emb.","encoder.blocks.",'
            '"decoder.autoencoder.encoder."]'
        ),
        "key_mapping": (
            '{"decoder.autoencoder.encoder.":"projector.",'
            '"encoder.blocks.":"encoder.blocks.",'
            '"encoder.tok_emb.":"encoder.tok_emb."}'
        ),
        "n_layer": "1",
        "n_head": "2",
        "n_embd": "4",
        "max_len": "12",
        "num_feats": "4",
        "vocab_name": "smi-ted-regex-v1",
        "vocab_size": "8",
        "vocab_sha256": (
            "8576b60e838336837f9e894457ef144f440c4d31bc8ceda2315fb5b28a6dfd95"
        ),
        "state_tensor_count": str(tiny_model_entry["state_tensor_count"]),
        "state_tensor_bytes": str(tiny_model_entry["state_tensor_bytes"]),
    }


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda state, metadata: state.pop(next(iter(state))), "missing keys"),
        (
            lambda state, metadata: state.update(
                {"unexpected.weight": torch.zeros(1)}
            ),
            "unexpected keys",
        ),
        (
            lambda state, metadata: metadata.update(format_version="2"),
            "format_version",
        ),
        (
            lambda state, metadata: metadata.update(vocab_size="999"),
            "vocab_size",
        ),
    ],
)
def test_strict_loader_rejects_incompatible_asset(
    tmp_path,
    tiny_model_entry,
    tiny_model_factory,
    tiny_vocab_path,
    mutation,
    expected,
):
    from tennetsac.smi_ted_light.asset_contract import (
        expected_safetensors_metadata,
    )
    from tennetsac.smi_ted_light.inference import load_smi_ted_inference

    model = tiny_model_factory()
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    metadata = expected_safetensors_metadata(tiny_model_entry)
    mutation(state, metadata)
    checkpoint = tmp_path / "invalid.safetensors"
    save_file(state, checkpoint, metadata=metadata)

    with pytest.raises(ValueError, match=expected):
        load_smi_ted_inference(checkpoint, tiny_vocab_path, tiny_model_entry)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (
            lambda state: state.update(
                {"encoder.tok_emb.weight": state["encoder.tok_emb.weight"].flatten()}
            ),
            "shape",
        ),
        (
            lambda state: state.update(
                {"encoder.tok_emb.weight": state["encoder.tok_emb.weight"].double()}
            ),
            "dtype",
        ),
    ],
)
def test_strict_loader_rejects_tensor_contract_mismatch(
    tmp_path,
    tiny_model_entry,
    tiny_model_factory,
    tiny_vocab_path,
    mutation,
    expected,
):
    from tennetsac.smi_ted_light.asset_contract import (
        expected_safetensors_metadata,
    )
    from tennetsac.smi_ted_light.inference import load_smi_ted_inference

    state = {
        name: tensor.clone()
        for name, tensor in tiny_model_factory().state_dict().items()
    }
    mutation(state)
    checkpoint = tmp_path / "invalid-tensor.safetensors"
    save_file(
        state,
        checkpoint,
        metadata=expected_safetensors_metadata(tiny_model_entry),
    )

    with pytest.raises(ValueError, match=expected):
        load_smi_ted_inference(checkpoint, tiny_vocab_path, tiny_model_entry)


def test_strict_loader_rejects_declared_tensor_byte_mismatch(
    tmp_path, tiny_model_entry, tiny_model_factory, tiny_vocab_path
):
    from tennetsac.smi_ted_light.asset_contract import (
        expected_safetensors_metadata,
    )
    from tennetsac.smi_ted_light.inference import load_smi_ted_inference

    incompatible_entry = dict(tiny_model_entry)
    incompatible_entry["state_tensor_bytes"] += 1
    state = {
        name: tensor.clone()
        for name, tensor in tiny_model_factory().state_dict().items()
    }
    checkpoint = tmp_path / "invalid-bytes.safetensors"
    save_file(
        state,
        checkpoint,
        metadata=expected_safetensors_metadata(incompatible_entry),
    )

    with pytest.raises(ValueError, match="state tensor bytes"):
        load_smi_ted_inference(checkpoint, tiny_vocab_path, incompatible_entry)


def test_strict_loader_loads_valid_safetensors_without_torch_load(
    tmp_path,
    tiny_model_entry,
    tiny_model_factory,
    tiny_vocab_path,
    monkeypatch,
):
    from tennetsac.smi_ted_light.asset_contract import (
        expected_safetensors_metadata,
    )
    from tennetsac.smi_ted_light.inference import (
        SmiTedInferenceModel,
        load_smi_ted_inference,
    )

    expected_state = {
        name: tensor.clone()
        for name, tensor in tiny_model_factory().state_dict().items()
    }
    checkpoint = tmp_path / "valid.safetensors"
    save_file(
        expected_state,
        checkpoint,
        metadata=expected_safetensors_metadata(tiny_model_entry),
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("torch.load must not be called")
        ),
    )

    loaded = load_smi_ted_inference(
        checkpoint, tiny_vocab_path, tiny_model_entry
    )

    assert isinstance(loaded, SmiTedInferenceModel)
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, expected_state[name])


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
