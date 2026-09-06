"""Validate release archives contain the package bundle and no repository debris."""

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "tennetsac.egg-info",
    "dist",
}
FORBIDDEN_SUFFIXES = {".pyc", ".so"}
WHEEL_FORBIDDEN_ROOTS = {"tests", "docs", ".github", "examples", "scripts"}
SDIST_FORBIDDEN_FILES = {
    "TeNNet-SAC.yml",
    "requirements.txt",
    "TrainingSystemsList.csv",
    "CITATION.bib",
    "CITATION.ris",
    "architecture.png",
}
SMI_TED_CHECKPOINT = "smi-ted-Light_40.pt"
CHECKPOINTS = {
    "base.ckpt",
    "geo.ckpt",
    "prf.ckpt",
    *(f"fine-tuned/{index}.ckpt" for index in range(1, 11)),
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
    required.add("METADATA" if kind == "wheel" else "PKG-INFO")
    return required


def _has_required_member(kind: str, members: set[str], required: str) -> bool:
    return required in members


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
        elif path.name == SMI_TED_CHECKPOINT:
            errors.append(f"forbidden SMI-TED checkpoint: {member}")
        elif kind == "wheel" and parts and parts[0] in WHEEL_FORBIDDEN_ROOTS:
            errors.append(f"forbidden archive member: {member}")
        elif kind == "sdist" and (
            (parts and parts[0] in WHEEL_FORBIDDEN_ROOTS)
            or path.name in SDIST_FORBIDDEN_FILES
        ):
            errors.append(f"forbidden archive member: {member}")
    return errors


def verify_archive(path: Path) -> list[str]:
    """Return every validation failure for a wheel or source distribution."""
    path = Path(path)
    root_errors = []
    if path.suffix == ".whl":
        kind = "wheel"
        try:
            with zipfile.ZipFile(path) as archive:
                members, path_errors = _normalized_members(archive.namelist())
        except (FileNotFoundError, zipfile.BadZipFile) as error:
            return [f"cannot read wheel: {error}"]
    else:
        kind = "sdist"
        try:
            with tarfile.open(path, "r:*") as archive:
                raw_members = [member.name for member in archive.getmembers()]
                normalized_members, path_errors = _normalized_members(raw_members)
                members, root_errors = _sdist_members(
                    normalized_members
                )
        except (FileNotFoundError, tarfile.TarError) as error:
            return [f"cannot read sdist: {error}"]

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
        if not _has_required_member(kind, member_set, required):
            errors.append(f"missing required member: {required}")
    return errors


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
