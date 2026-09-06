"""Verify release tags and distribution metadata agree exactly."""

import argparse
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

from packaging.version import InvalidVersion, Version


def version_from_tag(tag: str) -> str:
    """Return the canonical version for a v-prefixed release tag."""
    if not tag.startswith("v") or tag == "v":
        raise ValueError(f"release tag must start with v followed by a version: {tag}")
    raw_version = tag[1:]
    try:
        version = Version(raw_version)
    except InvalidVersion as error:
        raise ValueError(f"invalid release version in tag {tag}") from error
    if version.local is not None:
        raise ValueError(f"local versions are not allowed in release tags: {tag}")
    canonical = str(version)
    if canonical != raw_version:
        raise ValueError(
            f"release tag version must be canonical: {tag} should be v{canonical}"
        )
    return canonical


def _wheel_metadata(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_members = [
            name
            for name in archive.namelist()
            if (parts := PurePosixPath(name).parts)
            and len(parts) == 2
            and parts[1] == "METADATA"
            and parts[0].startswith("tennetsac-")
            and parts[0].endswith(".dist-info")
        ]
        if len(metadata_members) != 1:
            raise ValueError("expected exactly one tennetsac dist-info METADATA member")
        return archive.read(metadata_members[0]).decode("utf-8")


def _sdist_metadata(path: Path) -> str:
    with tarfile.open(path, "r:*") as archive:
        metadata_members = [
            member
            for member in archive.getmembers()
            if (
                (parts := PurePosixPath(member.name).parts)
                and len(parts) == 2
                and parts[1] == "PKG-INFO"
            )
        ]
        if len(metadata_members) != 1:
            raise ValueError("expected exactly one top-level PKG-INFO member")
        extracted = archive.extractfile(metadata_members[0])
        if extracted is None:
            raise ValueError("cannot read PKG-INFO member")
        return extracted.read().decode("utf-8")


def _artifact_version(path: Path) -> str:
    if path.suffix == ".whl":
        metadata = _wheel_metadata(path)
    elif path.name.endswith(".tar.gz"):
        metadata = _sdist_metadata(path)
    else:
        raise ValueError("expected a wheel or .tar.gz source distribution")
    version = Parser().parsestr(metadata).get("Version")
    if not version:
        raise ValueError("missing Version metadata")
    return version


def verify_release_version(tag: str, artifacts) -> list[str]:
    """Return every artifact version error for the release tag."""
    try:
        expected = version_from_tag(tag)
    except ValueError as error:
        return [str(error)]

    errors = []
    for artifact in artifacts:
        path = Path(artifact)
        try:
            actual = _artifact_version(path)
        except (OSError, tarfile.TarError, zipfile.BadZipFile, ValueError) as error:
            errors.append(f"{path}: {error}")
            continue
        if actual != expected:
            errors.append(
                f"{path}: artifact version {actual} does not match tag version {expected}"
            )
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("artifacts", nargs="+")
    args = parser.parse_args(argv)
    errors = verify_release_version(args.tag, args.artifacts)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
