"""Validate release archives contain an exact, byte-valid model bundle."""

import argparse
from email import policy
from email.parser import BytesParser
import hashlib
import importlib.util
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tennetsac"
    / "_manifest_schema.py"
)
_SCHEMA_SPEC = importlib.util.spec_from_file_location(
    "_tennetsac_manifest_schema", _SCHEMA_PATH
)
if _SCHEMA_SPEC is None or _SCHEMA_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load manifest schema validator from {_SCHEMA_PATH}")
_SCHEMA_MODULE = importlib.util.module_from_spec(_SCHEMA_SPEC)
_SCHEMA_SPEC.loader.exec_module(_SCHEMA_MODULE)
validate_manifest = _SCHEMA_MODULE.validate_manifest

FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "tennetsac.egg-info",
    "dist",
}
FORBIDDEN_SUFFIXES = {".pyc", ".so"}
FORBIDDEN_MODEL_WEIGHT_SUFFIXES = {".pt", ".safetensors"}
WHEEL_FORBIDDEN_ROOTS = {"tests", "docs", ".github", "examples", "scripts"}
SDIST_FORBIDDEN_FILES = {
    "TeNNet-SAC.yml",
    "requirements.txt",
    "TrainingSystemsList.csv",
    "CITATION.bib",
    "CITATION.ris",
    "architecture.png",
}
CHECKPOINTS = {
    "base.ckpt",
    "geo.ckpt",
    "prf.ckpt",
    *(f"fine-tuned/{index}.ckpt" for index in range(1, 11)),
}
LICENSE_EXPRESSION = "MIT AND Apache-2.0"
LICENSE_FILES = {
    "LICENSE",
    "licenses/IBM-materials-APACHE-2.0.txt",
    "licenses/fast-transformers-MIT.txt",
    "THIRD_PARTY_NOTICES.md",
}


def _normalized_member(name: str) -> tuple[str, list[str]]:
    """Return a portable archive path and errors that must survive normalization."""
    member = name.replace("\\", "/")
    errors = []
    if not member:
        errors.append("unsafe archive member (empty path)")
    elif member.startswith("/"):
        errors.append(f"unsafe archive member (absolute path): {name}")
    elif re.match(r"^[A-Za-z]:", member):
        errors.append(f"unsafe archive member (drive-qualified path): {name}")
    if ".." in member.split("/"):
        errors.append(f"unsafe archive member (parent traversal): {name}")
    return str(PurePosixPath(member)), errors


def _normalized_members(names: list[str]) -> tuple[list[str], list[str]]:
    members = []
    errors = []
    for name in names:
        member, member_errors = _normalized_member(name)
        members.append(member)
        errors.extend(member_errors)
    return members, errors


def _sdist_members(paths: list[str]) -> tuple[list[str], list[str]]:
    """Remove per-member root directories while reporting malformed root layouts."""
    roots = {PurePosixPath(path).parts[0] for path in paths if path not in {"", "."}}
    errors = []
    if len(roots) != 1:
        errors.append("sdist must use one top-level directory")
    stripped = []
    for path in paths:
        parts = PurePosixPath(path).parts
        stripped.append(str(PurePosixPath(*parts[1:])) if parts else path)
    return stripped, errors


def _required_members(kind: str) -> set[str]:
    package_root = "tennetsac" if kind == "wheel" else "src/tennetsac"
    required = {
        f"{package_root}/model_manifest.json",
        f"{package_root}/smi_ted_light/bert_vocab_curated.txt",
        *(f"{package_root}/ckpt_files/{name}" for name in CHECKPOINTS),
    }
    if kind == "sdist":
        required.update(LICENSE_FILES)
    required.add("METADATA" if kind == "wheel" else "PKG-INFO")
    return required


def _wheel_metadata_members(members: set[str]) -> list[str]:
    return [
        member
        for member in members
        if (parts := PurePosixPath(member).parts)
        and len(parts) == 2
        and parts[1] == "METADATA"
        and re.fullmatch(r"tennetsac-.+\.dist-info", parts[0])
    ]


def _inspect_members(kind: str, members: list[str]) -> list[str]:
    errors = []
    for member in members:
        path = PurePosixPath(member)
        parts = path.parts
        generated_sdist_metadata = (
            kind == "sdist"
            and member
            in {"src/tennetsac.egg-info", "src/tennetsac.egg-info/SOURCES.txt"}
        )
        if ".." in parts:
            errors.append(f"unsafe archive member: {member}")
        elif FORBIDDEN_PARTS.intersection(parts) and not generated_sdist_metadata:
            errors.append(f"forbidden archive member: {member}")
        elif path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden archive member: {member}")
        elif path.suffix.lower() in FORBIDDEN_MODEL_WEIGHT_SUFFIXES:
            errors.append(f"forbidden external model weight: {member}")
        elif kind == "wheel" and parts and parts[0] in WHEEL_FORBIDDEN_ROOTS:
            errors.append(f"forbidden archive member: {member}")
        elif kind == "sdist" and (
            (parts and parts[0] in WHEEL_FORBIDDEN_ROOTS)
            or path.name in SDIST_FORBIDDEN_FILES
        ):
            errors.append(f"forbidden archive member: {member}")
    return errors


def _metadata_errors(payload: bytes) -> list[str]:
    metadata = BytesParser(policy=policy.default).parsebytes(payload)
    errors = []
    if metadata.get("License-Expression") != LICENSE_EXPRESSION:
        errors.append(
            f"License-Expression must be {LICENSE_EXPRESSION!r}; "
            f"got {metadata.get('License-Expression')!r}"
        )
    license_files = metadata.get_all("License-File", [])
    if len(license_files) != len(set(license_files)) or set(license_files) != LICENSE_FILES:
        errors.append(
            f"License-File metadata must be exactly {sorted(LICENSE_FILES)}; "
            f"got {sorted(license_files)}"
        )
    return errors


def _archive_content_errors(kind, members, read_member) -> list[str]:
    errors = []
    member_set = set(members)
    package_root = "tennetsac" if kind == "wheel" else "src/tennetsac"
    manifest_member = f"{package_root}/model_manifest.json"

    metadata_members = (
        _wheel_metadata_members(member_set) if kind == "wheel" else ["PKG-INFO"]
    )
    if len(metadata_members) == 1 and metadata_members[0] in member_set:
        try:
            errors.extend(_metadata_errors(read_member(metadata_members[0])))
        except (KeyError, OSError) as error:
            errors.append(f"cannot read package metadata: {error}")

    if kind == "wheel" and len(metadata_members) == 1:
        metadata_root = str(PurePosixPath(metadata_members[0]).parent)
        for relative_path in sorted(LICENSE_FILES):
            member = f"{metadata_root}/licenses/{relative_path}"
            if member not in member_set:
                errors.append(f"missing required member: {member}")

    if manifest_member not in member_set:
        return errors
    try:
        manifest = json.loads(read_member(manifest_member).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        errors.append(f"model manifest is not valid JSON: {error}")
        return errors
    except (KeyError, OSError) as error:
        errors.append(f"cannot read model manifest: {error}")
        return errors

    schema_errors = validate_manifest(manifest)
    errors.extend(f"model manifest schema: {error}" for error in schema_errors)
    if schema_errors:
        return errors

    declared = {entry["path"]: entry["sha256"] for entry in manifest["artifacts"]}
    for relative_path, expected_digest in sorted(declared.items()):
        member = f"{package_root}/{relative_path}"
        if member not in member_set:
            errors.append(f"missing bundled artifact: {relative_path}")
            continue
        try:
            actual_digest = hashlib.sha256(read_member(member)).hexdigest()
        except (KeyError, OSError) as error:
            errors.append(f"cannot read bundled artifact {relative_path}: {error}")
            continue
        if actual_digest != expected_digest:
            errors.append(f"sha256 mismatch: {relative_path}")

    packaged_checkpoints = set()
    checkpoint_prefix = f"{package_root}/"
    for member in members:
        if PurePosixPath(member).suffix.lower() != ".ckpt":
            continue
        if member.startswith(checkpoint_prefix):
            packaged_checkpoints.add(member[len(checkpoint_prefix) :])
        else:
            packaged_checkpoints.add(member)
    for relative_path in sorted(packaged_checkpoints - set(declared)):
        errors.append(f"undeclared packaged checkpoint: {relative_path}")
    return errors


def _verify_open_archive(kind, raw_members, read_raw_member) -> list[str]:
    normalized_members, path_errors = _normalized_members(raw_members)
    root_errors = []
    if kind == "sdist":
        members, root_errors = _sdist_members(normalized_members)
    else:
        members = normalized_members

    source_by_member = {}
    for source, member in zip(raw_members, members):
        if member in source_by_member:
            path_errors.append(f"duplicate archive member: {member}")
        else:
            source_by_member[member] = source

    def read_member(member):
        return read_raw_member(source_by_member[member])

    errors = [*path_errors, *root_errors, *_inspect_members(kind, members)]
    member_set = set(members)
    for required in sorted(_required_members(kind)):
        if required == "METADATA":
            metadata_members = _wheel_metadata_members(member_set)
            if not metadata_members:
                errors.append("missing required member: METADATA")
            elif len(metadata_members) != 1:
                errors.append(
                    "expected exactly one tennetsac dist-info METADATA member"
                )
            continue
        if required not in member_set:
            errors.append(f"missing required member: {required}")
    errors.extend(_archive_content_errors(kind, members, read_member))
    return errors


def verify_archive(path: Path) -> list[str]:
    """Return every validation failure for a wheel or source distribution."""
    path = Path(path)
    if path.suffix == ".whl":
        try:
            with zipfile.ZipFile(path) as archive:
                return _verify_open_archive(
                    "wheel", archive.namelist(), archive.read
                )
        except (FileNotFoundError, zipfile.BadZipFile) as error:
            return [f"cannot read wheel: {error}"]

    try:
        with tarfile.open(path, "r:*") as archive:
            raw_members = [member.name for member in archive.getmembers()]

            def read_tar_member(name):
                stream = archive.extractfile(name)
                if stream is None:
                    raise OSError(f"{name} is not a regular file")
                return stream.read()

            return _verify_open_archive("sdist", raw_members, read_tar_member)
    except (FileNotFoundError, tarfile.TarError) as error:
        return [f"cannot read sdist: {error}"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+")
    args = parser.parse_args(argv)
    errors = [
        f"{archive}: {error}"
        for archive in map(Path, args.archives)
        for error in verify_archive(archive)
    ]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
