import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.verify_distribution import main, verify_archive


CHECKPOINTS = {
    "base.ckpt",
    "geo.ckpt",
    "prf.ckpt",
    *(f"fine-tuned/{index}.ckpt" for index in range(1, 11)),
}


def _wheel_members():
    yield "tennetsac/model_manifest.json"
    yield "tennetsac/smi_ted_light/bert_vocab_curated.txt"
    yield from (f"tennetsac/ckpt_files/{name}" for name in CHECKPOINTS)
    yield "tennetsac-0.1.10.dist-info/METADATA"


def _sdist_members():
    root = "tennetsac-0.1.10"
    yield f"{root}/src/tennetsac/model_manifest.json"
    yield f"{root}/src/tennetsac/smi_ted_light/bert_vocab_curated.txt"
    yield from (f"{root}/src/tennetsac/ckpt_files/{name}" for name in CHECKPOINTS)
    yield f"{root}/PKG-INFO"


def _write_wheel(path: Path, members=(), required_members=None):
    if required_members is None:
        required_members = _wheel_members()
    with zipfile.ZipFile(path, "w") as archive:
        for member in [*required_members, *members]:
            archive.writestr(member, "fixture")


def _write_sdist(path: Path, members=(), required_members=None):
    if required_members is None:
        required_members = _sdist_members()
    with tarfile.open(path, "w:gz") as archive:
        for member in [*required_members, *members]:
            payload = b"fixture"
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


@pytest.mark.parametrize(
    ("suffix", "writer", "bad_member"),
    [
        (".whl", _write_wheel, "tennetsac/__pycache__/runtime.pyc"),
        (".whl", _write_wheel, "tennetsac/native.so"),
        (".whl", _write_wheel, "tennetsac.egg-info/PKG-INFO"),
        (".whl", _write_wheel, "dist/tennetsac.whl"),
        (".whl", _write_wheel, "tests/test_runtime.py"),
        (".whl", _write_wheel, "docs/index.md"),
        (".whl", _write_wheel, ".github/workflows/ci.yml"),
        (".whl", _write_wheel, "examples/demo.py"),
        (".whl", _write_wheel, "scripts/release.py"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/src/tennetsac/__pycache__/runtime.pyc"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/src/tennetsac/native.so"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/tennetsac.egg-info/PKG-INFO"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/dist/tennetsac.whl"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/tests/test_runtime.py"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/docs/index.md"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/.github/workflows/ci.yml"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/examples/demo.py"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/scripts/release.py"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/TeNNet-SAC.yml"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/requirements.txt"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/TrainingSystemsList.csv"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/CITATION.bib"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/CITATION.ris"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/architecture.png"),
    ],
)
def test_verify_archive_rejects_forbidden_members(tmp_path, suffix, writer, bad_member):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    writer(archive, [bad_member])

    assert any(bad_member.rsplit("/", 1)[-1] in error or bad_member.split("/")[0] in error for error in verify_archive(archive))


@pytest.mark.parametrize(
    ("suffix", "writer", "required", "expected_member"),
    [
        *[
            (".whl", _write_wheel, member, member.rsplit("/", 1)[-1])
            if member.endswith(".dist-info/METADATA")
            else (".whl", _write_wheel, member, member)
            for member in _wheel_members()
        ],
        *[
            (".tar.gz", _write_sdist, member, member.split("/", 1)[1])
            for member in _sdist_members()
        ],
    ],
)
def test_verify_archive_requires_every_model_resource_and_metadata(
    tmp_path, suffix, writer, required, expected_member
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    members = list(_wheel_members() if suffix == ".whl" else _sdist_members())
    members.remove(required)
    writer(archive, required_members=members)

    errors = verify_archive(archive)

    assert any(expected_member in error for error in errors)


def test_wheel_rejects_unrelated_distribution_metadata(tmp_path):
    archive = tmp_path / "tennetsac-0.1.10.whl"
    members = [
        member for member in _wheel_members() if not member.endswith(".dist-info/METADATA")
    ]
    _write_wheel(
        archive,
        ["unrelated-1.0.dist-info/METADATA"],
        required_members=members,
    )

    assert "missing required member: METADATA" in verify_archive(archive)


def test_wheel_rejects_multiple_tennetsac_metadata_files(tmp_path):
    archive = tmp_path / "tennetsac-0.1.10.whl"
    _write_wheel(archive, ["tennetsac-2.0.dist-info/METADATA"])

    assert "expected exactly one tennetsac dist-info METADATA member" in verify_archive(
        archive
    )


@pytest.mark.parametrize(
    ("suffix", "writer", "bad_member", "reason"),
    [
        (".whl", _write_wheel, "/tests/escape.py", "absolute path"),
        (".whl", _write_wheel, "\\\\tests\\\\escape.py", "absolute path"),
        (".whl", _write_wheel, "C:\\tests\\escape.py", "drive-qualified path"),
        (".whl", _write_wheel, "tennetsac\\..\\tests\\escape.py", "parent traversal"),
        (".whl", _write_wheel, "", "empty path"),
        (".tar.gz", _write_sdist, "/tests/escape.py", "absolute path"),
        (".tar.gz", _write_sdist, "\\\\tests\\\\escape.py", "absolute path"),
        (".tar.gz", _write_sdist, "C:\\tests\\escape.py", "drive-qualified path"),
        (".tar.gz", _write_sdist, "root\\..\\tests\\escape.py", "parent traversal"),
        (".tar.gz", _write_sdist, "", "empty path"),
    ],
)
def test_verify_archive_rejects_unsafe_member_paths(
    tmp_path, suffix, writer, bad_member, reason
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    writer(archive, [bad_member])

    assert any(reason in error for error in verify_archive(archive))


def test_malformed_sdist_reports_root_forbidden_and_missing_errors(tmp_path):
    archive = tmp_path / "tennetsac-0.1.10.tar.gz"
    _write_sdist(
        archive,
        ["root-a/tests/test_runtime.py", "root-b/src/tennetsac/model_manifest.json"],
        required_members=[],
    )

    errors = verify_archive(archive)

    assert "sdist must use one top-level directory" in errors
    assert "forbidden archive member: tests/test_runtime.py" in errors
    assert "missing required member: src/tennetsac/smi_ted_light/bert_vocab_curated.txt" in errors


@pytest.mark.parametrize(
    ("suffix", "writer"),
    [(".whl", _write_wheel), (".tar.gz", _write_sdist)],
)
def test_verify_archive_accepts_complete_clean_distribution(tmp_path, suffix, writer):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    writer(archive)

    assert verify_archive(archive) == []


def test_verify_archive_allows_setuptools_generated_sdist_source_list(tmp_path):
    archive = tmp_path / "tennetsac-0.1.10.tar.gz"
    _write_sdist(
        archive,
        [
            "tennetsac-0.1.10/src/tennetsac.egg-info",
            "tennetsac-0.1.10/src/tennetsac.egg-info/SOURCES.txt",
        ],
    )

    assert verify_archive(archive) == []


def test_main_reports_every_archive_violation(tmp_path, capsys):
    archive = tmp_path / "tennetsac-0.1.10.whl"
    _write_wheel(archive, ["tests/test_runtime.py", "tennetsac/native.so"])

    assert main([str(archive)]) == 1

    output = capsys.readouterr().err
    assert "tests/test_runtime.py" in output
    assert "tennetsac/native.so" in output
