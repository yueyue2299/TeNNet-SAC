# Portions derived from https://github.com/IBM/materials
# Upstream commit: b16a458f37e6ce91997d3d3f6a12037971eb9f94
# Upstream path: models/smi_ted/inference/smi_ted_light/load.py
# License: Apache-2.0 (see licenses/IBM-materials-APACHE-2.0.txt)
# Modified for TeNNet-SAC; see THIRD_PARTY_NOTICES.md for the change summary.

from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from .asset_contract import (
    SMI_TED_LIGHT_CONTRACT,
    expected_safetensors_metadata,
)
from .fast_transformers.feature_maps import GeneralizedRandomFeatures
from .fast_transformers.masking import LengthMask
from .load import RotateEncoderBuilder, normalize_smiles
from .tokenizer import MolTranBertTokenizer


def production_config() -> dict[str, int | float]:
    config: dict[str, int | float] = dict(SMI_TED_LIGHT_CONTRACT.architecture)
    config["d_dropout"] = 0.2
    return config


class SmiTedInferenceEncoder(nn.Module):
    def __init__(self, config, n_vocab):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(n_vocab, config["n_embd"])
        self.drop = nn.Dropout(config["d_dropout"])

        builder = RotateEncoderBuilder.from_kwargs(
            n_layers=config["n_layer"],
            n_heads=config["n_head"],
            query_dimensions=config["n_embd"] // config["n_head"],
            value_dimensions=config["n_embd"] // config["n_head"],
            feed_forward_dimensions=config["n_embd"],
            attention_type="linear",
            feature_map=partial(
                GeneralizedRandomFeatures,
                n_dims=config["num_feats"],
                deterministic_eval=True,
            ),
            activation="gelu",
        )
        self.blocks = builder.get()

    def forward(self, idx, mask):
        x = self.tok_emb(idx)
        x = self.drop(x)
        x = self.blocks(
            x,
            length_mask=LengthMask(mask.sum(-1), max_len=idx.shape[1]),
        )

        input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        mask_embeddings = x * input_mask_expanded
        return F.pad(
            mask_embeddings,
            pad=(0, 0, 0, self.config["max_len"] - mask_embeddings.shape[1]),
            value=0,
        )


class SmiTedProjector(nn.Module):
    def __init__(self, feature_size, latent_size):
        super().__init__()
        self.fc1 = nn.Linear(feature_size, latent_size)
        self.ln_f = nn.LayerNorm(latent_size)
        self.lat = nn.Linear(latent_size, latent_size, bias=False)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.ln_f(x)
        return self.lat(x)


class SmiTedInferenceModel(nn.Module):
    def __init__(self, tokenizer, config=None):
        super().__init__()
        self.config = dict(config or production_config())
        self.tokenizer = tokenizer
        self.padding_idx = tokenizer.get_padding_idx()
        self.max_len = self.config["max_len"]
        self.n_embd = self.config["n_embd"]
        self.encoder = SmiTedInferenceEncoder(self.config, len(tokenizer.vocab))
        self.projector = SmiTedProjector(
            self.max_len * self.n_embd,
            self.n_embd,
        )

    def tokenize(self, smiles):
        batch = [smiles] if isinstance(smiles, str) else smiles
        tokens = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.max_len,
        )

        device = self.encoder.tok_emb.weight.device
        idx = tokens["input_ids"].clone().detach().to(device)
        mask = tokens["attention_mask"].clone().detach().to(device)
        return idx, mask

    def extract_embeddings(self, smiles):
        self.encoder.eval()
        idx, mask = self.tokenize(smiles)
        token_embeddings = self.encoder(idx, mask)
        embedding = self.projector(
            token_embeddings.reshape(-1, self.max_len * self.n_embd)
        )
        idx = F.pad(
            idx,
            pad=(0, self.max_len - idx.shape[1], 0, 0),
            value=self.padding_idx,
        )
        return idx, token_embeddings, embedding

    def encode(self, smiles, useCuda=False, batch_size=100, return_torch=False):
        del useCuda
        smiles = pd.Series(smiles) if isinstance(smiles, str) else pd.Series(list(smiles))
        smiles = smiles.apply(normalize_smiles)
        null_idx = smiles[smiles.isnull()].index.to_list()
        smiles = smiles.dropna()

        flat_list = []
        if not smiles.empty:
            n_split = (
                smiles.shape[0] // batch_size
                if smiles.shape[0] >= batch_size
                else smiles.shape[0]
            )
            embeddings = [
                self.extract_embeddings(list(batch))[2].cpu().detach().numpy()
                for batch in np.array_split(smiles.to_numpy(), n_split)
            ]
            flat_list = [item for batch in embeddings for item in batch]

        for idx in null_idx:
            flat_list.insert(idx, np.array([np.nan] * self.n_embd))
        result = np.asarray(flat_list)
        if result.size == 0:
            result = result.reshape(0, self.n_embd)

        if return_torch:
            return torch.tensor(result)
        return pd.DataFrame(result)


def _metadata_mismatch_message(actual, expected):
    mismatches = []
    for name in sorted(expected.keys() | actual.keys()):
        if name not in actual:
            mismatches.append(f"missing {name}")
        elif name not in expected:
            mismatches.append(f"unexpected {name}")
        elif actual[name] != expected[name]:
            mismatches.append(
                f"{name} expected {expected[name]!r}, got {actual[name]!r}"
            )
    return "; ".join(mismatches)


def load_smi_ted_inference(
    checkpoint_path, vocab_path, asset_entry
) -> SmiTedInferenceModel:
    expected_metadata = expected_safetensors_metadata(asset_entry)
    with safe_open(
        checkpoint_path, framework="pt", device="cpu"
    ) as checkpoint:
        actual_metadata = checkpoint.metadata() or {}
        if actual_metadata != expected_metadata:
            details = _metadata_mismatch_message(
                actual_metadata, expected_metadata
            )
            raise ValueError(f"safetensors metadata mismatch: {details}")
        actual_keys = set(checkpoint.keys())

    tokenizer = MolTranBertTokenizer(vocab_file=str(vocab_path))
    if len(tokenizer) != asset_entry["vocab_size"]:
        raise ValueError(
            "vocab_size mismatch: "
            f"expected {asset_entry['vocab_size']}, got {len(tokenizer)}"
        )

    config = dict(asset_entry["architecture"])
    config["d_dropout"] = 0.2
    model = SmiTedInferenceModel(tokenizer, config)
    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    if missing_keys:
        raise ValueError(f"safetensors missing keys: {missing_keys}")
    if unexpected_keys:
        raise ValueError(f"safetensors unexpected keys: {unexpected_keys}")
    if len(actual_keys) != asset_entry["state_tensor_count"]:
        raise ValueError(
            "safetensors state tensor count mismatch: "
            f"expected {asset_entry['state_tensor_count']}, got {len(actual_keys)}"
        )

    state = load_file(checkpoint_path, device="cpu")
    for name, expected_tensor in expected_state.items():
        actual_tensor = state[name]
        if actual_tensor.shape != expected_tensor.shape:
            raise ValueError(
                f"safetensors shape mismatch for {name}: "
                f"expected {tuple(expected_tensor.shape)}, "
                f"got {tuple(actual_tensor.shape)}"
            )
        if actual_tensor.dtype != expected_tensor.dtype:
            raise ValueError(
                f"safetensors dtype mismatch for {name}: "
                f"expected {expected_tensor.dtype}, got {actual_tensor.dtype}"
            )

    actual_tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in state.values()
    )
    if actual_tensor_bytes != asset_entry["state_tensor_bytes"]:
        raise ValueError(
            "safetensors state tensor bytes mismatch: "
            f"expected {asset_entry['state_tensor_bytes']}, "
            f"got {actual_tensor_bytes}"
        )

    model.load_state_dict(state, strict=True)
    return model
