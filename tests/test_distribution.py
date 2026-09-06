import io
import json
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
CHECKPOINT_PAYLOAD = b"fixture checkpoint\n"
CHECKPOINT_SHA256 = "472bb8a60bd4b125cdabf44b456e125f0d6851375d8d03c5c038a6be960a5e09"
ARTIFACT_NAMES = {
    "base.ckpt": "gamma-base",
    "geo.ckpt": "geometry",
    "prf.ckpt": "sigma-profile",
    **{
        f"fine-tuned/{index}.ckpt": f"gamma-tuned-{index}"
        for index in range(1, 11)
    },
}
LICENSE_FILES = [
    "LICENSE",
    "licenses/IBM-materials-APACHE-2.0.txt",
    "licenses/fast-transformers-MIT.txt",
    "THIRD_PARTY_NOTICES.md",
]


def _manifest_bytes():
    return json.dumps(
        {
            "schema_version": 1,
            "bundle_version": "1.0.0",
            "artifacts": [
                {
                    "name": ARTIFACT_NAMES[path],
                    "path": f"ckpt_files/{path}",
                    "sha256": CHECKPOINT_SHA256,
                    "distribution": "bundled",
                }
                for path in sorted(CHECKPOINTS)
            ],
            "external_models": [
                {
                    "name": "chemberta2",
                    "source": "DeepChem/ChemBERTa-77M-MLM",
                    "revision": "ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
                    "distribution": "external",
                },
                {
                    "name": "smi-ted-light",
                    "source": "ibm/materials.smi-ted",
                    "filename": "smi-ted-Light_40.pt",
                    "revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
                    "sha256": "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
                    "distribution": "external",
                },
            ],
            "tokenizers": [
                {
                    "name": "smi-ted-regex",
                    "format_version": 1,
                    "vocab_path": "smi_ted_light/bert_vocab_curated.txt",
                }
            ],
        },
        separators=(",", ":"),
    ).encode()


def _metadata_bytes():
    license_headers = "".join(f"License-File: {path}\n" for path in LICENSE_FILES)
    return (
        "Metadata-Version: 2.4\n"
        "Name: tennetsac\n"
        "Version: 0.1.10\n"
        "License-Expression: MIT AND Apache-2.0\n"
        f"{license_headers}\n"
    ).encode()


def _wheel_files():
    metadata_root = "tennetsac-0.1.10.dist-info"
    files = {
        "tennetsac/model_manifest.json": _manifest_bytes(),
        "tennetsac/smi_ted_light/bert_vocab_curated.txt": b"<bos>\n<eos>\n",
        f"{metadata_root}/METADATA": _metadata_bytes(),
        **{
            f"tennetsac/ckpt_files/{name}": CHECKPOINT_PAYLOAD
            for name in CHECKPOINTS
        },
    }
    files.update(
        {
            f"{metadata_root}/licenses/{path}": b"license fixture\n"
            for path in LICENSE_FILES
        }
    )
    return files


def _sdist_files():
    root = "tennetsac-0.1.10"
    files = {
        f"{root}/src/tennetsac/model_manifest.json": _manifest_bytes(),
        f"{root}/src/tennetsac/smi_ted_light/bert_vocab_curated.txt": b"<bos>\n<eos>\n",
        f"{root}/PKG-INFO": _metadata_bytes(),
        **{
            f"{root}/src/tennetsac/ckpt_files/{name}": CHECKPOINT_PAYLOAD
            for name in CHECKPOINTS
        },
    }
    files.update({f"{root}/{path}": b"license fixture\n" for path in LICENSE_FILES})
    return files


def _wheel_members():
    yield from _wheel_files()


def _sdist_members():
    yield from _sdist_files()


def _archive_files(base_files, members, required_members):
    files = (
        dict(base_files)
        if required_members is None
        else {member: base_files.get(member, b"fixture\n") for member in required_members}
    )
    if hasattr(members, "items"):
        files.update(members)
    else:
        files.update({member: b"fixture\n" for member in members})
    return files


def _write_wheel(path: Path, members=(), required_members=None):
    files = _archive_files(_wheel_files(), members, required_members)
    with zipfile.ZipFile(path, "w") as archive:
        for member, payload in files.items():
            archive.writestr(member, payload)


def _write_sdist(path: Path, members=(), required_members=None):
    files = _archive_files(_sdist_files(), members, required_members)
    with tarfile.open(path, "w:gz") as archive:
        for member, payload in files.items():
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


def test_verify_wheel_rejects_empty_member_path(tmp_path, monkeypatch):
    archive = tmp_path / "tennetsac-0.1.10.whl"
    archive.write_bytes(b"fixture")

    class FakeZipFile:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def namelist(self):
            return [*_wheel_members(), ""]

        def read(self, member):
            return _wheel_files()[member]

    monkeypatch.setattr(zipfile, "ZipFile", FakeZipFile)

    assert any("empty path" in error for error in verify_archive(archive))


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


@pytest.mark.parametrize(
    ("suffix", "writer", "manifest_member"),
    [
        (".whl", _write_wheel, "tennetsac/model_manifest.json"),
        (
            ".tar.gz",
            _write_sdist,
            "tennetsac-0.1.10/src/tennetsac/model_manifest.json",
        ),
    ],
)
def test_verify_archive_rejects_malformed_manifest_json(
    tmp_path, suffix, writer, manifest_member
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    writer(archive, {manifest_member: b"{not json"})

    assert any("model manifest is not valid JSON" in error for error in verify_archive(archive))


@pytest.mark.parametrize(
    ("suffix", "writer", "manifest_member"),
    [
        (".whl", _write_wheel, "tennetsac/model_manifest.json"),
        (
            ".tar.gz",
            _write_sdist,
            "tennetsac-0.1.10/src/tennetsac/model_manifest.json",
        ),
    ],
)
def test_verify_archive_rejects_invalid_manifest_schema(
    tmp_path, suffix, writer, manifest_member
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    manifest = json.loads(_manifest_bytes())
    manifest["artifacts"][0].pop("distribution")
    writer(archive, {manifest_member: json.dumps(manifest).encode()})

    assert any("artifact keys" in error for error in verify_archive(archive))


@pytest.mark.parametrize(
    ("suffix", "writer", "checkpoint_member"),
    [
        (".whl", _write_wheel, "tennetsac/ckpt_files/base.ckpt"),
        (
            ".tar.gz",
            _write_sdist,
            "tennetsac-0.1.10/src/tennetsac/ckpt_files/base.ckpt",
        ),
    ],
)
def test_verify_archive_rejects_checkpoint_digest_mismatch(
    tmp_path, suffix, writer, checkpoint_member
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    writer(archive, {checkpoint_member: b"corrupt checkpoint\n"})

    errors = verify_archive(archive)

    assert any("sha256 mismatch: ckpt_files/base.ckpt" in error for error in errors)


@pytest.mark.parametrize(
    ("suffix", "writer", "checkpoint_member"),
    [
        (".whl", _write_wheel, "tennetsac/ckpt_files/base.ckpt"),
        (
            ".tar.gz",
            _write_sdist,
            "tennetsac-0.1.10/src/tennetsac/ckpt_files/base.ckpt",
        ),
    ],
)
def test_verify_archive_rejects_missing_manifest_declared_checkpoint(
    tmp_path, suffix, writer, checkpoint_member
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    members = list(_wheel_members() if suffix == ".whl" else _sdist_members())
    members.remove(checkpoint_member)
    writer(archive, required_members=members)

    assert any(
        "missing bundled artifact: ckpt_files/base.ckpt" in error
        for error in verify_archive(archive)
    )


@pytest.mark.parametrize(
    ("suffix", "writer", "checkpoint_member"),
    [
        (".whl", _write_wheel, "tennetsac/ckpt_files/undeclared.ckpt"),
        (
            ".tar.gz",
            _write_sdist,
            "tennetsac-0.1.10/src/tennetsac/ckpt_files/undeclared.ckpt",
        ),
    ],
)
def test_verify_archive_rejects_undeclared_checkpoint(
    tmp_path, suffix, writer, checkpoint_member
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    writer(archive, {checkpoint_member: CHECKPOINT_PAYLOAD})

    assert any(
        "undeclared packaged checkpoint: ckpt_files/undeclared.ckpt" in error
        for error in verify_archive(archive)
    )


@pytest.mark.parametrize(
    ("suffix", "writer", "metadata_member"),
    [
        (".whl", _write_wheel, "tennetsac-0.1.10.dist-info/METADATA"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/PKG-INFO"),
    ],
)
def test_verify_archive_rejects_incomplete_license_metadata(
    tmp_path, suffix, writer, metadata_member
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    metadata = _metadata_bytes().replace(
        b"License-Expression: MIT AND Apache-2.0",
        b"License-Expression: MIT",
    )
    writer(archive, {metadata_member: metadata})

    assert any("License-Expression" in error for error in verify_archive(archive))


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
