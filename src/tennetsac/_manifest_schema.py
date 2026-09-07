"""Strict validation for the version-2 TeNNet-SAC model manifest."""

import re


TOP_LEVEL_KEYS = {
    "schema_version", "bundle_version", "artifacts", "external_models", "tokenizers"
}
ARTIFACT_KEYS = {"name", "path", "sha256", "distribution"}
BUNDLED_CHECKPOINT_PATHS = frozenset(
    {
        "ckpt_files/base.ckpt", "ckpt_files/geo.ckpt", "ckpt_files/prf.ckpt",
        *(f"ckpt_files/fine-tuned/{index}.ckpt" for index in range(1, 11)),
    }
)
CHEMBERTA2_KEYS = {"name", "source", "revision", "distribution"}
SMI_TED_LIGHT_KEYS = {
    "name", "distribution", "format", "format_version", "repository", "release_tag",
    "url", "filename", "sha256", "state_tensor_count", "state_tensor_bytes",
    "vocab_size", "architecture", "parent", "pruning", "legacy_override_env",
}
SMI_TED_ARCHITECTURE = {
    "n_layer": 12, "n_head": 12, "n_embd": 768, "max_len": 202, "num_feats": 32,
}
SMI_TED_PARENT = {
    "historical_repository": "ibm/materials.smi-ted",
    "canonical_repository": "ibm-research/materials.smi-ted",
    "revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
    "filename": "smi-ted-Light_40.pt",
    "sha256": "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
}
SMI_TED_PREFIXES = [
    "encoder.tok_emb.", "encoder.blocks.", "decoder.autoencoder.encoder.",
]
SMI_TED_REPOSITORY = "yueyue2299/TeNNet-SAC"
SMI_TED_RELEASE_TAG = "model-smi-ted-light-v1"
SMI_TED_FILENAME = "smi-ted-light-inference-v1.safetensors"
SMI_TED_URL = (
    "https://github.com/yueyue2299/TeNNet-SAC/releases/download/"
    "model-smi-ted-light-v1/smi-ted-light-inference-v1.safetensors"
)
TOKENIZER_KEYS = {"name", "format_version", "vocab_path"}
TOKENIZER_NAME = "smi-ted-regex"
TOKENIZER_VOCAB_PATH = "smi_ted_light/bert_vocab_curated.txt"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_BUNDLE_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


def _exact_keys(value, expected, label, errors):
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return False
    actual = set(value)
    if actual != expected:
        errors.append(
            f"{label} keys must be exactly {sorted(expected)}; got {sorted(actual)}"
        )
        return False
    return True


def _nonempty_string(value, label, errors):
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a non-empty string")
        return False
    return True


def _exact_integer(value, expected, label, errors):
    if type(value) is not int or value != expected:
        errors.append(f"{label} must be the integer {expected}")


def _validate_smi_ted_light(entry, errors):
    if entry["distribution"] != "external":
        errors.append("external model distribution for smi-ted-light must be external")
    if entry["format"] != "safetensors":
        errors.append("smi-ted-light format must be safetensors")
    _exact_integer(entry["format_version"], 1, "smi-ted-light format_version", errors)
    if entry["repository"] != SMI_TED_REPOSITORY:
        errors.append(f"smi-ted-light repository must be {SMI_TED_REPOSITORY}")
    if entry["release_tag"] != SMI_TED_RELEASE_TAG:
        errors.append(f"smi-ted-light release_tag must be {SMI_TED_RELEASE_TAG}")
    if entry["filename"] != SMI_TED_FILENAME:
        errors.append(f"smi-ted-light filename must be {SMI_TED_FILENAME}")
    if entry["url"] != SMI_TED_URL:
        errors.append("smi-ted-light URL must be the immutable HTTPS GitHub URL")
    if not isinstance(entry["sha256"], str) or not _SHA256.fullmatch(entry["sha256"]):
        errors.append("smi-ted-light sha256 must be lowercase 64-hex")
    _exact_integer(entry["state_tensor_count"], 224, "smi-ted-light state_tensor_count", errors)
    _exact_integer(entry["state_tensor_bytes"], 656641536, "smi-ted-light state_tensor_bytes", errors)
    _exact_integer(entry["vocab_size"], 2393, "smi-ted-light vocab_size", errors)

    architecture = entry["architecture"]
    if _exact_keys(architecture, set(SMI_TED_ARCHITECTURE), "smi-ted-light architecture", errors):
        for field, expected in SMI_TED_ARCHITECTURE.items():
            _exact_integer(architecture[field], expected, f"smi-ted-light architecture {field}", errors)

    parent = entry["parent"]
    if _exact_keys(parent, set(SMI_TED_PARENT), "smi-ted-light parent", errors):
        for field, expected in SMI_TED_PARENT.items():
            if parent[field] != expected:
                errors.append(f"smi-ted-light parent {field} must be {expected}")

    pruning = entry["pruning"]
    if _exact_keys(pruning, {"rule_version", "included_prefixes"}, "smi-ted-light pruning", errors):
        _exact_integer(pruning["rule_version"], 1, "smi-ted-light pruning rule_version", errors)
        if pruning["included_prefixes"] != SMI_TED_PREFIXES:
            errors.append("smi-ted-light pruning included_prefixes must be the approved ordered list")
    if entry["legacy_override_env"] != "TENNETSAC_SMI_TED_CHECKPOINT":
        errors.append("smi-ted-light legacy_override_env must be TENNETSAC_SMI_TED_CHECKPOINT")


def validate_manifest(manifest) -> list[str]:
    """Return every version-2 release-manifest schema violation."""
    errors = []
    if not _exact_keys(manifest, TOP_LEVEL_KEYS, "top-level", errors):
        if not isinstance(manifest, dict):
            return errors

    _exact_integer(manifest.get("schema_version"), 2, "schema_version", errors)
    bundle_version = manifest.get("bundle_version")
    if not isinstance(bundle_version, str) or not _BUNDLE_VERSION.fullmatch(bundle_version):
        errors.append("bundle_version must use MAJOR.MINOR.PATCH")

    artifacts = manifest.get("artifacts")
    artifact_names = set()
    artifact_paths = set()
    if not isinstance(artifacts, list):
        errors.append("artifacts must be a list")
    else:
        for index, entry in enumerate(artifacts):
            label = f"artifact keys at index {index}"
            if not _exact_keys(entry, ARTIFACT_KEYS, label, errors):
                continue
            name = entry["name"]
            path = entry["path"]
            if _nonempty_string(name, f"artifact name at index {index}", errors):
                if name in artifact_names:
                    errors.append(f"duplicate artifact name: {name}")
                artifact_names.add(name)
            if _nonempty_string(path, f"artifact path at index {index}", errors):
                if path in artifact_paths:
                    errors.append(f"duplicate artifact path: {path}")
                artifact_paths.add(path)
            if not isinstance(entry["sha256"], str) or not _SHA256.fullmatch(entry["sha256"]):
                errors.append(f"artifact sha256 at index {index} must be lowercase 64-hex")
            if entry["distribution"] != "bundled":
                errors.append(f"artifact distribution at index {index} must be bundled")
        if artifact_paths != BUNDLED_CHECKPOINT_PATHS:
            errors.append("checkpoint paths must be exactly the approved 13-checkpoint set")

    external_models = manifest.get("external_models")
    external_names = set()
    if not isinstance(external_models, list):
        errors.append("external_models must be a list")
    else:
        for index, entry in enumerate(external_models):
            if not isinstance(entry, dict):
                errors.append(f"external model at index {index} must be an object")
                continue
            name = entry.get("name")
            expected_keys = {"chemberta2": CHEMBERTA2_KEYS, "smi-ted-light": SMI_TED_LIGHT_KEYS}.get(name)
            if expected_keys is None:
                errors.append(f"unknown external model at index {index}: {name!r}")
                continue
            if not _exact_keys(entry, expected_keys, f"external model keys for {name}", errors):
                continue
            if name in external_names:
                errors.append(f"duplicate external model name: {name}")
            external_names.add(name)
            if name == "chemberta2":
                _nonempty_string(entry["source"], "external model source for chemberta2", errors)
                if not isinstance(entry["revision"], str) or not _REVISION.fullmatch(entry["revision"]):
                    errors.append("external model revision for chemberta2 must be lowercase 40-hex")
                if entry["distribution"] != "external":
                    errors.append("external model distribution for chemberta2 must be external")
            else:
                _validate_smi_ted_light(entry, errors)
        if external_names != {"chemberta2", "smi-ted-light"}:
            errors.append("external model names must be exactly chemberta2 and smi-ted-light")

    tokenizers = manifest.get("tokenizers")
    if not isinstance(tokenizers, list):
        errors.append("tokenizers must be a list")
    elif len(tokenizers) != 1:
        errors.append("tokenizers must contain exactly one entry")
    else:
        tokenizer = tokenizers[0]
        if _exact_keys(tokenizer, TOKENIZER_KEYS, "tokenizer keys", errors):
            if tokenizer["name"] != TOKENIZER_NAME:
                errors.append(f"tokenizer name must be {TOKENIZER_NAME}")
            _exact_integer(tokenizer["format_version"], 1, "tokenizer format_version", errors)
            if tokenizer["vocab_path"] != TOKENIZER_VOCAB_PATH:
                errors.append(f"tokenizer vocab_path must be {TOKENIZER_VOCAB_PATH}")

    return errors
