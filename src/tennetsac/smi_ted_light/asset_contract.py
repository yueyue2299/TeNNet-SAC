import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class SmiTedAssetContract:
    parent_repository_historical: str
    parent_repository_canonical: str
    parent_revision: str
    parent_filename: str
    parent_sha256: str
    architecture: Mapping[str, int]
    vocab_name: str
    vocab_size: int
    vocab_sha256: str
    prefix_map: Mapping[str, str]
    allowed_excluded_prefixes: tuple[str, ...]
    tensor_count: int
    tensor_bytes: int
    parameter_bytes: int
    buffer_bytes: int


SMI_TED_LIGHT_CONTRACT = SmiTedAssetContract(
    parent_repository_historical="ibm/materials.smi-ted",
    parent_repository_canonical="ibm-research/materials.smi-ted",
    parent_revision="414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
    parent_filename="smi-ted-Light_40.pt",
    parent_sha256="baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
    architecture=MappingProxyType(
        {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "max_len": 202,
            "num_feats": 32,
        }
    ),
    vocab_name="smi-ted-regex-v1",
    vocab_size=2393,
    vocab_sha256="8576b60e838336837f9e894457ef144f440c4d31bc8ceda2315fb5b28a6dfd95",
    prefix_map=MappingProxyType(
        {
            "encoder.tok_emb.": "encoder.tok_emb.",
            "encoder.blocks.": "encoder.blocks.",
            "decoder.autoencoder.encoder.": "projector.",
        }
    ),
    allowed_excluded_prefixes=(
        "encoder.lang_model.",
        "decoder.autoencoder.decoder.",
        "decoder.lang_model.",
    ),
    tensor_count=224,
    tensor_bytes=656_641_536,
    parameter_bytes=656_541_696,
    buffer_bytes=99_840,
)


def expected_safetensors_metadata(asset_entry) -> dict[str, str]:
    return {
        "format": "tennetsac-smi-ted-inference",
        "format_version": "1",
        "upstream_repository": (
            SMI_TED_LIGHT_CONTRACT.parent_repository_canonical
        ),
        "upstream_revision": SMI_TED_LIGHT_CONTRACT.parent_revision,
        "upstream_filename": SMI_TED_LIGHT_CONTRACT.parent_filename,
        "upstream_sha256": SMI_TED_LIGHT_CONTRACT.parent_sha256,
        "pruning_rule_version": "1",
        "included_prefixes": json.dumps(
            list(SMI_TED_LIGHT_CONTRACT.prefix_map), separators=(",", ":")
        ),
        "key_mapping": json.dumps(
            dict(SMI_TED_LIGHT_CONTRACT.prefix_map),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "n_layer": str(asset_entry["architecture"]["n_layer"]),
        "n_head": str(asset_entry["architecture"]["n_head"]),
        "n_embd": str(asset_entry["architecture"]["n_embd"]),
        "max_len": str(asset_entry["architecture"]["max_len"]),
        "num_feats": str(asset_entry["architecture"]["num_feats"]),
        "vocab_name": SMI_TED_LIGHT_CONTRACT.vocab_name,
        "vocab_size": str(asset_entry["vocab_size"]),
        "vocab_sha256": SMI_TED_LIGHT_CONTRACT.vocab_sha256,
        "state_tensor_count": str(asset_entry["state_tensor_count"]),
        "state_tensor_bytes": str(asset_entry["state_tensor_bytes"]),
    }
