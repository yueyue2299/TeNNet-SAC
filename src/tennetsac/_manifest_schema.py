"""Lightweight validation for the version-1 TeNNet-SAC model manifest."""

import re


TOP_LEVEL_KEYS = {
    "schema_version",
    "bundle_version",
    "artifacts",
    "external_models",
    "tokenizers",
}
ARTIFACT_KEYS = {"name", "path", "sha256", "distribution"}
BUNDLED_CHECKPOINT_PATHS = frozenset(
    {
        "ckpt_files/base.ckpt",
        "ckpt_files/geo.ckpt",
        "ckpt_files/prf.ckpt",
        *(f"ckpt_files/fine-tuned/{index}.ckpt" for index in range(1, 11)),
    }
)
EXTERNAL_MODEL_KEYS = {
    "chemberta2": {"name", "source", "revision", "distribution"},
    "smi-ted-light": {
        "name",
        "source",
        "filename",
        "revision",
        "sha256",
        "distribution",
    },
}
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


def validate_manifest(manifest) -> list[str]:
    """Return every version-1 release-manifest schema violation."""
    errors = []
    if not _exact_keys(manifest, TOP_LEVEL_KEYS, "top-level", errors):
        if not isinstance(manifest, dict):
            return errors

    if type(manifest.get("schema_version")) is not int or manifest.get(
        "schema_version"
    ) != 1:
        errors.append("schema_version must be the integer 1")

    bundle_version = manifest.get("bundle_version")
    if not isinstance(bundle_version, str) or not _BUNDLE_VERSION.fullmatch(
        bundle_version
    ):
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
            if not isinstance(entry["sha256"], str) or not _SHA256.fullmatch(
                entry["sha256"]
            ):
                errors.append(f"artifact sha256 at index {index} must be lowercase 64-hex")
            if entry["distribution"] != "bundled":
                errors.append(f"artifact distribution at index {index} must be bundled")
        if artifact_paths != BUNDLED_CHECKPOINT_PATHS:
            errors.append(
                "checkpoint paths must be exactly the approved 13-checkpoint set"
            )

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
            expected_keys = EXTERNAL_MODEL_KEYS.get(name)
            if expected_keys is None:
                errors.append(f"unknown external model at index {index}: {name!r}")
                continue
            if not _exact_keys(
                entry, expected_keys, f"external model keys for {name}", errors
            ):
                continue
            if name in external_names:
                errors.append(f"duplicate external model name: {name}")
            external_names.add(name)
            _nonempty_string(entry["source"], f"external model source for {name}", errors)
            if not isinstance(entry["revision"], str) or not _REVISION.fullmatch(
                entry["revision"]
            ):
                errors.append(
                    f"external model revision for {name} must be lowercase 40-hex"
                )
            if entry["distribution"] != "external":
                errors.append(f"external model distribution for {name} must be external")
            if "filename" in entry:
                _nonempty_string(
                    entry["filename"], f"external model filename for {name}", errors
                )
            if "sha256" in entry and (
                not isinstance(entry["sha256"], str)
                or not _SHA256.fullmatch(entry["sha256"])
            ):
                errors.append(
                    f"external model sha256 for {name} must be lowercase 64-hex"
                )
        if external_names != set(EXTERNAL_MODEL_KEYS):
            errors.append(
                "external model names must be exactly chemberta2 and smi-ted-light"
            )

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
            if (
                type(tokenizer["format_version"]) is not int
                or tokenizer["format_version"] != 1
            ):
                errors.append("tokenizer format_version must be the integer 1")
            if tokenizer["vocab_path"] != TOKENIZER_VOCAB_PATH:
                errors.append(f"tokenizer vocab_path must be {TOKENIZER_VOCAB_PATH}")

    return errors
