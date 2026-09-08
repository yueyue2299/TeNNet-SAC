import hashlib
import json
from importlib.resources import as_file, files

from ._manifest_schema import validate_manifest


def load_manifest() -> dict:
    resource = files("tennetsac").joinpath("model_manifest.json")
    manifest = json.loads(resource.read_text(encoding="utf-8"))
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Invalid model manifest: " + "; ".join(errors))
    return manifest


def external_model(name: str) -> dict:
    for entry in load_manifest()["external_models"]:
        if entry["name"] == name:
            return entry
    raise KeyError(f"Unknown external model: {name}")


def bundled_artifact(name: str) -> dict:
    for entry in load_manifest()["artifacts"]:
        if entry["name"] == name:
            return entry
    raise KeyError(f"Unknown bundled artifact: {name}")


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundled_artifacts() -> list[str]:
    errors = []
    package_root = files("tennetsac")
    for entry in load_manifest()["artifacts"]:
        resource = package_root.joinpath(entry["path"])
        if not resource.is_file():
            errors.append(f"missing: {entry['path']}")
            continue
        with as_file(resource) as path:
            digest = _sha256(path)
        if digest != entry["sha256"]:
            errors.append(f"sha256 mismatch: {entry['path']}")
    return errors
