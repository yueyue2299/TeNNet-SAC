import copy
import re

import pytest

from tennetsac import model_manifest


EXPECTED_ARTIFACTS = [
    {"name": "gamma-base", "path": "ckpt_files/base.ckpt", "sha256": "791ce5bf9c59e2099467882b4e7ee220575f464f2f3e007495a78ed9ae6f2d93", "distribution": "bundled"},
    {"name": "geometry", "path": "ckpt_files/geo.ckpt", "sha256": "ab4e37731eb07573cf4d9a7a15c906c6ef8b0ae5a895ff4f921a736c32483624", "distribution": "bundled"},
    {"name": "sigma-profile", "path": "ckpt_files/prf.ckpt", "sha256": "649e7139cc43a95bd459d0eb0c58fac3a5d3a62a9b15ae6bdb0018894c42723e", "distribution": "bundled"},
    {"name": "gamma-tuned-1", "path": "ckpt_files/fine-tuned/1.ckpt", "sha256": "134be46cce6839a7b08250750e4803c8198a4725d5973ca124db76dd488aedc1", "distribution": "bundled"},
    {"name": "gamma-tuned-2", "path": "ckpt_files/fine-tuned/2.ckpt", "sha256": "d763688bbe2898fc182ae6e9b39436a678ad892510c76b510c43c389e87d4c5b", "distribution": "bundled"},
    {"name": "gamma-tuned-3", "path": "ckpt_files/fine-tuned/3.ckpt", "sha256": "15232df78dbbce274818e00bdfff455a163fbb9ff8af1ef3ba4de73650519400", "distribution": "bundled"},
    {"name": "gamma-tuned-4", "path": "ckpt_files/fine-tuned/4.ckpt", "sha256": "bf01d2c6832b161ddf12d789dc886a3c3065a1810948d196f103a89c94f0daeb", "distribution": "bundled"},
    {"name": "gamma-tuned-5", "path": "ckpt_files/fine-tuned/5.ckpt", "sha256": "937a02283521bba254b917685f4db265c48617732f9c5bbfffb14eaaf1d38813", "distribution": "bundled"},
    {"name": "gamma-tuned-6", "path": "ckpt_files/fine-tuned/6.ckpt", "sha256": "7588bfe7a265752395373cf930ede2e721da2cb37fd7779a7b3b94d6c3590e79", "distribution": "bundled"},
    {"name": "gamma-tuned-7", "path": "ckpt_files/fine-tuned/7.ckpt", "sha256": "f4c03f8281b7ac56213e5e04123c208fc4f2030ca610684bd1cbd9aa59f7e16d", "distribution": "bundled"},
    {"name": "gamma-tuned-8", "path": "ckpt_files/fine-tuned/8.ckpt", "sha256": "0ff319a81281b4315638281283433590e6adf7b7ab17039919601747f736096d", "distribution": "bundled"},
    {"name": "gamma-tuned-9", "path": "ckpt_files/fine-tuned/9.ckpt", "sha256": "73c65639135717a3eaa7130c03ea6f85fa3f95ed1b5937b5b73611bd33378c70", "distribution": "bundled"},
    {"name": "gamma-tuned-10", "path": "ckpt_files/fine-tuned/10.ckpt", "sha256": "def4bb97274011564b33d55666bcc5d4d2bcae9c422cbf2c96529c7c364ebfcc", "distribution": "bundled"},
]
EXPECTED_EXTERNAL_MODELS = [
    {
        "name": "chemberta2",
        "source": "DeepChem/ChemBERTa-77M-MLM",
        "revision": "ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
        "distribution": "external",
    },
    {
        "name": "smi-ted-light",
        "distribution": "external",
        "format": "safetensors",
        "format_version": 1,
        "repository": "yueyue2299/TeNNet-SAC",
        "release_tag": "model-smi-ted-light-v1",
        "url": "https://github.com/yueyue2299/TeNNet-SAC/releases/download/model-smi-ted-light-v1/smi-ted-light-inference-v1.safetensors",
        "filename": "smi-ted-light-inference-v1.safetensors",
        "sha256": "1eda6afcb37fcaa85c6303ed57decf0b84294f9bb8ccc79538f9e1c22701acf4",
        "state_tensor_count": 224,
        "state_tensor_bytes": 656641536,
        "vocab_size": 2393,
        "architecture": {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "max_len": 202,
            "num_feats": 32,
        },
        "parent": {
            "historical_repository": "ibm/materials.smi-ted",
            "canonical_repository": "ibm-research/materials.smi-ted",
            "revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
            "filename": "smi-ted-Light_40.pt",
            "sha256": "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
        },
        "pruning": {
            "rule_version": 1,
            "included_prefixes": [
                "encoder.tok_emb.",
                "encoder.blocks.",
                "decoder.autoencoder.encoder.",
            ],
        },
        "legacy_override_env": "TENNETSAC_SMI_TED_CHECKPOINT",
    },
]
EXPECTED_TOKENIZERS = [
    {
        "name": "smi-ted-regex",
        "format_version": 1,
        "vocab_path": "smi_ted_light/bert_vocab_curated.txt",
    }
]


def test_manifest_matches_the_complete_literal_release_schema() -> None:
    manifest = model_manifest.load_manifest()

    assert set(manifest) == {
        "schema_version",
        "bundle_version",
        "artifacts",
        "external_models",
        "tokenizers",
    }
    assert manifest["schema_version"] == 2
    assert manifest["bundle_version"] == "1.0.0"
    assert manifest["artifacts"] == EXPECTED_ARTIFACTS
    assert manifest["external_models"] == EXPECTED_EXTERNAL_MODELS
    assert manifest["tokenizers"] == EXPECTED_TOKENIZERS
    assert all(
        set(entry) == {"name", "path", "sha256", "distribution"}
        for entry in manifest["artifacts"]
    )
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        for entry in manifest["artifacts"]
    )
    assert model_manifest.validate_manifest(manifest) == []
    assert model_manifest.verify_bundled_artifacts() == []


def test_manifest_pins_external_model_sources_and_revisions() -> None:
    for expected in EXPECTED_EXTERNAL_MODELS:
        assert model_manifest.external_model(expected["name"]) == expected


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda manifest: manifest.update(schema_version=1), "schema_version"),
        (lambda manifest: manifest.update(unexpected=True), "top-level keys"),
        (
            lambda manifest: manifest["artifacts"][0].pop("distribution"),
            "artifact keys",
        ),
        (
            lambda manifest: manifest["artifacts"][0].update(distribution="external"),
            "distribution",
        ),
        (
            lambda manifest: manifest["artifacts"][0].update(sha256="A" * 64),
            "sha256",
        ),
        (
            lambda manifest: manifest["artifacts"][1].update(
                name=manifest["artifacts"][0]["name"]
            ),
            "duplicate artifact name",
        ),
        (
            lambda manifest: manifest["artifacts"][1].update(
                path=manifest["artifacts"][0]["path"]
            ),
            "duplicate artifact path",
        ),
        (
            lambda manifest: manifest["artifacts"][0].update(
                path="ckpt_files/unapproved.ckpt"
            ),
            "checkpoint paths",
        ),
        (
            lambda manifest: manifest["external_models"][0].update(revision="main"),
            "revision",
        ),
        (
            lambda manifest: manifest["external_models"][1].pop("filename"),
            "external model keys",
        ),
        (
            lambda manifest: manifest["external_models"][1].update(
                url="http://github.com/yueyue2299/TeNNet-SAC/releases/download/model-smi-ted-light-v1/smi-ted-light-inference-v1.safetensors"
            ),
            "immutable HTTPS GitHub URL",
        ),
        (
            lambda manifest: manifest["external_models"][1].update(
                release_tag="model-smi-ted-light-v2"
            ),
            "release_tag",
        ),
        (
            lambda manifest: manifest["external_models"][1].update(
                state_tensor_count=223
            ),
            "state_tensor_count",
        ),
        (
            lambda manifest: manifest["external_models"][1]["architecture"].update(
                n_embd=512
            ),
            "architecture",
        ),
        (
            lambda manifest: manifest["external_models"][1]["parent"].update(
                canonical_repository="ibm/materials.smi-ted"
            ),
            "parent",
        ),
        (
            lambda manifest: manifest["external_models"][1]["pruning"].update(
                included_prefixes=[
                    "encoder.blocks.",
                    "encoder.tok_emb.",
                    "decoder.autoencoder.encoder.",
                ]
            ),
            "included_prefixes",
        ),
        (
            lambda manifest: manifest["external_models"][1].update(
                legacy_override_env="TENNETSAC_OTHER_CHECKPOINT"
            ),
            "legacy_override_env",
        ),
        (
            lambda manifest: manifest["tokenizers"][0].update(
                path=manifest["tokenizers"][0].pop("vocab_path")
            ),
            "tokenizer keys",
        ),
    ],
)
def test_manifest_validator_rejects_incomplete_or_unapproved_schema(
    mutation, expected_error
) -> None:
    manifest = copy.deepcopy(
        {
            "schema_version": 2,
            "bundle_version": "1.0.0",
            "artifacts": EXPECTED_ARTIFACTS,
            "external_models": EXPECTED_EXTERNAL_MODELS,
            "tokenizers": EXPECTED_TOKENIZERS,
        }
    )
    mutation(manifest)

    validator = getattr(model_manifest, "validate_manifest", None)
    assert callable(validator), "model manifest schema validator is missing"
    assert any(expected_error in error for error in validator(manifest))
