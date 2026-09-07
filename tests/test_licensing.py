import hashlib
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    from setuptools._vendor import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
LICENSE_FILES = {
    "licenses/IBM-materials-APACHE-2.0.txt": (
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    ),
    "licenses/fast-transformers-MIT.txt": (
        "db2b1cdf9bda73650d860da722095b8f3a817be008e94e3522bad2c40979915d"
    ),
}


def test_vendored_upstream_license_texts_are_verbatim() -> None:
    for relative_path, expected_digest in LICENSE_FILES.items():
        payload = (ROOT / relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_digest


def test_pep639_metadata_declares_and_packages_every_license_file() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["license"] == "MIT AND Apache-2.0"
    assert project["license-files"] == [
        "LICENSE",
        "licenses/IBM-materials-APACHE-2.0.txt",
        "licenses/fast-transformers-MIT.txt",
        "THIRD_PARTY_NOTICES.md",
    ]


def test_third_party_notice_records_immutable_sources_and_local_scope() -> None:
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    normalized_notice = " ".join(notice.split())

    required_facts = {
        "https://github.com/IBM/materials",
        "b16a458f37e6ce91997d3d3f6a12037971eb9f94",
        "models/smi_ted/inference/smi_ted_light/",
        "Apache-2.0",
        "https://github.com/idiap/fast-transformers",
        "2ad36b97e64cb93862937bd21fcc9568d989561f",
        "C++",
        "CUDA",
        "package-relative imports",
        "tokenizer",
        "digest checks",
        "actionable error handling",
    }
    assert not sorted(fact for fact in required_facts if fact not in normalized_notice)


def test_third_party_notice_records_smi_ted_derived_asset_provenance() -> None:
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    normalized_notice = " ".join(notice.split())

    required_facts = {
        "ibm-research/materials.smi-ted",
        "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
        "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
        "smi-ted-light-inference-v1.safetensors",
        "pruned",
        "224 tensors",
        "Apache-2.0",
        "TeNNet-SAC-distributed",
        "not an IBM-published checkpoint",
        "does not imply IBM endorsement",
    }

    assert not sorted(fact for fact in required_facts if fact not in normalized_notice)


def test_modified_apache_sources_carry_provenance_headers() -> None:
    for filename in ("load.py", "tokenizer.py"):
        header = "\n".join(
            (ROOT / "src" / "tennetsac" / "smi_ted_light" / filename)
            .read_text(encoding="utf-8")
            .splitlines()[:12]
        )
        assert "https://github.com/IBM/materials" in header
        assert "b16a458f37e6ce91997d3d3f6a12037971eb9f94" in header
        assert "Apache-2.0" in header
        assert "modified" in header.lower()
