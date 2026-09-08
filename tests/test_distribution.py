import io
import hashlib
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.verify_distribution import main, verify_archive


ROOT = Path(__file__).resolve().parents[1]
MODEL_ASSETS = {
    "base.ckpt": "gamma-base",
    "geo.ckpt": "geometry",
    "prf.ckpt": "sigma-profile",
    "fine-tuned/gamma-ensemble-v1.safetensors": "gamma-tuned-ensemble",
}
MODEL_PAYLOADS = {
    path: f"fixture model asset: {path}\n".encode() for path in MODEL_ASSETS
}
MODEL_SHA256 = {
    path: hashlib.sha256(payload).hexdigest()
    for path, payload in MODEL_PAYLOADS.items()
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
            "schema_version": 2,
            "bundle_version": "2.0.0",
            "artifacts": [
                {
                    "name": MODEL_ASSETS[path],
                    "path": f"ckpt_files/{path}",
                    "sha256": MODEL_SHA256[path],
                    "distribution": "bundled",
                }
                for path in sorted(MODEL_ASSETS)
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
                    "distribution": "external",
                    "format": "safetensors",
                    "format_version": 1,
                    "repository": "yueyue2299/TeNNet-SAC",
                    "release_tag": "model-smi-ted-light-v1",
                    "url": "https://github.com/yueyue2299/TeNNet-SAC/releases/download/model-smi-ted-light-v1/smi-ted-light-inference-v1.safetensors",
                    "filename": "smi-ted-light-inference-v1.safetensors",
                    "sha256": "566c828ab592a4bfd9050906e4d7f64273a9a27517bef175b00dbed38eec94fc",
                    "state_tensor_count": 224,
                    "state_tensor_bytes": 656641536,
                    "vocab_size": 2393,
                    "architecture": {
                        "n_layer": 12,
                        "n_head": 12,
                        "n_embd": 768,
                        "max_len": 202,
                        "num_feats": 32,
                    },
                    "parent": {
                        "historical_repository": "ibm/materials.smi-ted",
                        "canonical_repository": "ibm-research/materials.smi-ted",
                        "revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
                        "filename": "smi-ted-Light_40.pt",
                        "sha256": "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
                    },
                    "pruning": {
                        "rule_version": 1,
                        "included_prefixes": [
                            "encoder.tok_emb.",
                            "encoder.blocks.",
                            "decoder.autoencoder.encoder.",
                        ],
                    },
                    "legacy_override_env": "TENNETSAC_SMI_TED_CHECKPOINT",
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
            f"tennetsac/ckpt_files/{name}": MODEL_PAYLOADS[name]
            for name in MODEL_ASSETS
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
            f"{root}/src/tennetsac/ckpt_files/{name}": MODEL_PAYLOADS[name]
            for name in MODEL_ASSETS
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
    ("suffix", "writer", "bad_member"),
    [
        (".whl", _write_wheel, "tennetsac/models/smi-ted-Light_40.pt"),
        (".whl", _write_wheel, "tennetsac/models/smi-ted-light-inference-v1.safetensors"),
        (".tar.gz", _write_sdist, "tennetsac-0.1.10/assets/smi-ted-Light_40.pt"),
        (
            ".tar.gz",
            _write_sdist,
            "tennetsac-0.1.10/assets/smi-ted-light-inference-v1.safetensors",
        ),
    ],
)
def test_verify_archive_rejects_external_model_weight_suffixes(
    tmp_path, suffix, writer, bad_member
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    writer(archive, [bad_member])

    assert any(
        "forbidden external model weight" in error and bad_member.rsplit("/", 1)[-1] in error
        for error in verify_archive(archive)
    )


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


def test_real_sdist_includes_the_pinned_gamma_ensemble_asset(tmp_path):
    """Breaks if source-manifest exclusions drop the one approved safetensors file."""
    output = tmp_path / "dist"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--outdir",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    archive = next(output.glob("*.tar.gz"))
    with tarfile.open(archive, "r:gz") as source_distribution:
        members = source_distribution.getnames()
    assert any(
        member.endswith("src/tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors")
        for member in members
    )
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
    ("suffix", "writer", "model_member", "relative_path"),
    [
        (
            ".whl",
            _write_wheel,
            "tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors",
            "ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors",
        ),
        (
            ".tar.gz",
            _write_sdist,
            "tennetsac-0.1.10/src/tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors",
            "ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors",
        ),
    ],
)
def test_verify_archive_rejects_model_weight_digest_mismatch(
    tmp_path, suffix, writer, model_member, relative_path
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    writer(archive, {model_member: b"corrupt model weight\n"})

    errors = verify_archive(archive)

    assert f"sha256 mismatch: {relative_path}" in errors


@pytest.mark.parametrize(
    ("suffix", "writer", "model_member", "relative_path"),
    [
        (
            ".whl",
            _write_wheel,
            "tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors",
            "ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors",
        ),
        (
            ".tar.gz",
            _write_sdist,
            "tennetsac-0.1.10/src/tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors",
            "ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors",
        ),
    ],
)
def test_verify_archive_rejects_missing_manifest_declared_model_weight(
    tmp_path, suffix, writer, model_member, relative_path
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    members = list(_wheel_members() if suffix == ".whl" else _sdist_members())
    members.remove(model_member)
    writer(archive, required_members=members)

    assert any(
        f"missing bundled artifact: {relative_path}" in error
        for error in verify_archive(archive)
    )


@pytest.mark.parametrize(
    ("suffix", "writer", "model_weight_member", "relative_path"),
    [
        (
            ".whl",
            _write_wheel,
            "tennetsac/ckpt_files/fine-tuned/7.ckpt",
            "ckpt_files/fine-tuned/7.ckpt",
        ),
        (
            ".tar.gz",
            _write_sdist,
            "tennetsac-0.1.10/src/tennetsac/ckpt_files/fine-tuned/7.ckpt",
            "ckpt_files/fine-tuned/7.ckpt",
        ),
    ],
)
def test_verify_archive_rejects_old_numbered_checkpoint(
    tmp_path, suffix, writer, model_weight_member, relative_path
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    writer(archive, {model_weight_member: MODEL_PAYLOADS["base.ckpt"]})

    assert any(
        f"undeclared packaged model weight: {relative_path}" in error
        for error in verify_archive(archive)
    )


@pytest.mark.parametrize(
    ("suffix", "writer", "model_weight_member", "relative_path"),
    [
        (
            ".whl",
            _write_wheel,
            "tennetsac/ckpt_files/fine-tuned/other.safetensors",
            "ckpt_files/fine-tuned/other.safetensors",
        ),
        (
            ".tar.gz",
            _write_sdist,
            "tennetsac-0.1.10/src/tennetsac/ckpt_files/fine-tuned/other.safetensors",
            "ckpt_files/fine-tuned/other.safetensors",
        ),
    ],
)
def test_verify_archive_rejects_undeclared_safetensors_weight(
    tmp_path, suffix, writer, model_weight_member, relative_path
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    writer(archive, {model_weight_member: MODEL_PAYLOADS["base.ckpt"]})

    errors = verify_archive(archive)

    assert any(
        error.startswith("forbidden external model weight: ")
        and error.endswith(relative_path)
        for error in errors
    )
    assert f"undeclared packaged model weight: {relative_path}" in errors


@pytest.mark.parametrize(
    ("suffix", "writer", "model_weight_member", "relative_path"),
    [
        (
            ".whl",
            _write_wheel,
            "tennetsac/ckpt_files/fine-tuned/unapproved.pt",
            "ckpt_files/fine-tuned/unapproved.pt",
        ),
        (
            ".tar.gz",
            _write_sdist,
            "tennetsac-0.1.10/src/tennetsac/ckpt_files/fine-tuned/unapproved.pt",
            "ckpt_files/fine-tuned/unapproved.pt",
        ),
    ],
)
def test_verify_archive_rejects_all_pt_model_weights(
    tmp_path, suffix, writer, model_weight_member, relative_path
):
    archive = tmp_path / f"tennetsac-0.1.10{suffix}"
    writer(archive, {model_weight_member: MODEL_PAYLOADS["base.ckpt"]})

    assert any(
        error.startswith("forbidden external model weight: ")
        and error.endswith(relative_path)
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
