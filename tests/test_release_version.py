import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from packaging.version import Version


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "verify_release_version.py"
ROOT = Path(__file__).parents[1]


def _release_module():
    if not SCRIPT_PATH.exists():
        pytest.fail("missing scripts/verify_release_version.py")
    spec = importlib.util.spec_from_file_location("verify_release_version", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_wheel(path: Path, version: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"tennetsac-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: tennetsac\nVersion: {version}\n",
        )
    return path


def _write_sdist(path: Path, version: str) -> Path:
    package_root = f"tennetsac-{version}"
    metadata = f"Metadata-Version: 2.4\nName: tennetsac\nVersion: {version}\n"
    metadata_path = Path("PKG-INFO")
    tmp_metadata = path.parent / metadata_path
    tmp_metadata.write_text(metadata, encoding="utf-8")
    with tarfile.open(path, "w:gz") as archive:
        archive.add(tmp_metadata, arcname=f"{package_root}/PKG-INFO")
    tmp_metadata.unlink()
    return path


def _write_sdist_with_egg_info(path: Path, version: str) -> Path:
    package_root = f"tennetsac-{version}"
    top_metadata = path.parent / "PKG-INFO"
    egg_metadata = path.parent / "EGG-PKG-INFO"
    metadata = f"Metadata-Version: 2.4\nName: tennetsac\nVersion: {version}\n"
    top_metadata.write_text(metadata, encoding="utf-8")
    egg_metadata.write_text(
        metadata.replace("Name: tennetsac", "Name: ignored"),
        encoding="utf-8",
    )
    with tarfile.open(path, "w:gz") as archive:
        archive.add(top_metadata, arcname=f"{package_root}/PKG-INFO")
        archive.add(
            egg_metadata,
            arcname=f"{package_root}/src/tennetsac.egg-info/PKG-INFO",
        )
    top_metadata.unlink()
    egg_metadata.unlink()
    return path


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("v0.2.0", "0.2.0"), ("v1.2.3rc1", "1.2.3rc1")],
)
def test_version_from_tag(tag, expected):
    module = _release_module()

    assert module.version_from_tag(tag) == expected


@pytest.mark.parametrize("tag", ["0.2.0", "release-v1", "vnext", "v1.0+local"])
def test_invalid_release_tag_is_rejected(tag):
    module = _release_module()

    with pytest.raises(ValueError):
        module.version_from_tag(tag)


def test_setuptools_scm_ignores_model_release_tags(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    shutil.copy(ROOT / "pyproject.toml", repository / "pyproject.toml")

    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "ci@example.invalid")
    git("config", "user.name", "CI")
    git("config", "commit.gpgsign", "false")
    git("config", "tag.gpgSign", "false")
    git("add", "pyproject.toml")
    git("commit", "-m", "initial")
    git("tag", "v0.2.0")

    marker = repository / "marker.txt"
    marker.write_text("model release\n", encoding="utf-8")
    git("add", "marker.txt")
    git("commit", "-m", "model release")
    git("tag", "model-smi-ted-light-v9")

    marker.write_text("development\n", encoding="utf-8")
    git("add", "marker.txt")
    git("commit", "-m", "development")

    result = subprocess.run(
        [sys.executable, "-m", "setuptools_scm"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    version = Version(result.stdout.strip())

    assert version.release == (0, 2, 0)
    assert version.post == 1
    assert version.dev == 2


def test_verify_release_version_accepts_matching_wheel_and_sdist(tmp_path):
    module = _release_module()
    wheel = _write_wheel(tmp_path / "tennetsac-0.2.0-py3-none-any.whl", "0.2.0")
    sdist = _write_sdist(tmp_path / "tennetsac-0.2.0.tar.gz", "0.2.0")

    assert module.verify_release_version("v0.2.0", [wheel, sdist]) == []


def test_verify_release_version_reads_top_level_sdist_metadata(tmp_path):
    module = _release_module()
    sdist = _write_sdist_with_egg_info(tmp_path / "tennetsac-0.2.0.tar.gz", "0.2.0")

    assert module.verify_release_version("v0.2.0", [sdist]) == []


def test_verify_release_version_rejects_wheel_version_mismatch(tmp_path):
    module = _release_module()
    wheel = _write_wheel(tmp_path / "tennetsac-0.2.1-py3-none-any.whl", "0.2.1")

    errors = module.verify_release_version("v0.2.0", [wheel])

    assert errors == [
        f"{wheel}: artifact version 0.2.1 does not match tag version 0.2.0"
    ]


def test_verify_sha256sums_accepts_exact_distribution_set(tmp_path):
    module = _release_module()
    wheel = _write_wheel(tmp_path / "tennetsac-0.2.0-py3-none-any.whl", "0.2.0")
    sdist = _write_sdist(tmp_path / "tennetsac-0.2.0.tar.gz", "0.2.0")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n"
        f"{hashlib.sha256(sdist.read_bytes()).hexdigest()}  {sdist.name}\n",
        encoding="utf-8",
    )

    assert module.verify_sha256sums_manifest(manifest, [wheel, sdist]) == []


def test_verify_sha256sums_rejects_unlisted_distribution(tmp_path):
    module = _release_module()
    wheel = _write_wheel(tmp_path / "tennetsac-0.2.0-py3-none-any.whl", "0.2.0")
    sdist = _write_sdist(tmp_path / "tennetsac-0.2.0.tar.gz", "0.2.0")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n",
        encoding="utf-8",
    )

    assert module.verify_sha256sums_manifest(manifest, [wheel, sdist]) == [
        f"{manifest}: missing checksum entries for {sdist.name}"
    ]


def test_verify_sha256sums_rejects_extra_or_unsafe_entries(tmp_path):
    module = _release_module()
    wheel = _write_wheel(tmp_path / "tennetsac-0.2.0-py3-none-any.whl", "0.2.0")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n"
        f"{'0' * 64}  extra-0.2.0.tar.gz\n"
        f"{'1' * 64}  nested/tennetsac-0.2.0.tar.gz\n",
        encoding="utf-8",
    )

    assert module.verify_sha256sums_manifest(manifest, [wheel]) == [
        f"{manifest}: unsafe checksum entry name nested/tennetsac-0.2.0.tar.gz",
        f"{manifest}: checksum entries without downloaded distributions: "
        "extra-0.2.0.tar.gz, nested/tennetsac-0.2.0.tar.gz",
    ]
