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


def _write_wheel(path: Path, members=()):
    with zipfile.ZipFile(path, "w") as archive:
        for member in [*_wheel_members(), *members]:
            archive.writestr(member, "fixture")


def _write_sdist(path: Path, members=()):
    with tarfile.open(path, "w:gz") as archive:
        for member in [*_sdist_members(), *members]:
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
    ("suffix", "writer", "expected_member"),
    [
        (".whl", _write_wheel, "tennetsac/model_manifest.json"),
        (".tar.gz", _write_sdist, "src/tennetsac/model_manifest.json"),
    ],
)
def test_verify_archive_requires_model_bundle_and_metadata(
    tmp_path, suffix, writer, expected_member
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    writer(archive)
    if suffix == ".whl":
        members = list(_wheel_members())
    else:
        members = list(_sdist_members())
    members.remove(members[[member.endswith(expected_member) for member in members].index(True)])
    if suffix == ".whl":
        with zipfile.ZipFile(archive, "w") as built:
            for member in members:
                built.writestr(member, "fixture")
    else:
        with tarfile.open(archive, "w:gz") as built:
            for member in members:
                payload = b"fixture"
                info = tarfile.TarInfo(member)
                info.size = len(payload)
                built.addfile(info, io.BytesIO(payload))

    errors = verify_archive(archive)

    assert any(expected_member in error for error in errors)


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
