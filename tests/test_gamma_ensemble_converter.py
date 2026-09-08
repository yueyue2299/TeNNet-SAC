import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from scripts import build_gamma_ensemble_asset as converter
from tennetsac.models.GammaEnsemble import GammaEnsemble
from tennetsac.models.Prf2Gamma import Prf_to_Seg_Model


EXPECTED_SOURCE_CONTRACT = {
    "format_version": 1,
    "source_bundle_version": "1.0.0",
    "source_commit": "7711c6bd3dff7e2d7cff00590062f9e1a0ca605a",
    "members": [
        {
            "member": 1,
            "path": "ckpt_files/fine-tuned/1.ckpt",
            "sha256": "134be46cce6839a7b08250750e4803c8198a4725d5973ca124db76dd488aedc1",
        },
        {
            "member": 2,
            "path": "ckpt_files/fine-tuned/2.ckpt",
            "sha256": "d763688bbe2898fc182ae6e9b39436a678ad892510c76b510c43c389e87d4c5b",
        },
        {
            "member": 3,
            "path": "ckpt_files/fine-tuned/3.ckpt",
            "sha256": "15232df78dbbce274818e00bdfff455a163fbb9ff8af1ef3ba4de73650519400",
        },
        {
            "member": 4,
            "path": "ckpt_files/fine-tuned/4.ckpt",
            "sha256": "bf01d2c6832b161ddf12d789dc886a3c3065a1810948d196f103a89c94f0daeb",
        },
        {
            "member": 5,
            "path": "ckpt_files/fine-tuned/5.ckpt",
            "sha256": "937a02283521bba254b917685f4db265c48617732f9c5bbfffb14eaaf1d38813",
        },
        {
            "member": 6,
            "path": "ckpt_files/fine-tuned/6.ckpt",
            "sha256": "7588bfe7a265752395373cf930ede2e721da2cb37fd7779a7b3b94d6c3590e79",
        },
        {
            "member": 7,
            "path": "ckpt_files/fine-tuned/7.ckpt",
            "sha256": "f4c03f8281b7ac56213e5e04123c208fc4f2030ca610684bd1cbd9aa59f7e16d",
        },
        {
            "member": 8,
            "path": "ckpt_files/fine-tuned/8.ckpt",
            "sha256": "0ff319a81281b4315638281283433590e6adf7b7ab17039919601747f736096d",
        },
        {
            "member": 9,
            "path": "ckpt_files/fine-tuned/9.ckpt",
            "sha256": "73c65639135717a3eaa7130c03ea6f85fa3f95ed1b5937b5b73611bd33378c70",
        },
        {
            "member": 10,
            "path": "ckpt_files/fine-tuned/10.ckpt",
            "sha256": "def4bb97274011564b33d55666bcc5d4d2bcae9c422cbf2c96529c7c364ebfcc",
        },
    ],
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_members(count):
    torch.manual_seed(4107)
    first = Prf_to_Seg_Model().eval()
    members = [first]
    for seed in range(4108, 4108 + count - 1):
        torch.manual_seed(seed)
        member = Prf_to_Seg_Model().eval()
        state = member.state_dict()
        for key, value in first.state_dict().items():
            if not key.startswith("model_final."):
                state[key].copy_(value)
        members.append(member)
    return members


def write_legacy_sources(tmp_path, count=10):
    sources = tmp_path / "sources"
    sources.mkdir()
    members = _legacy_members(count)
    records = []
    for number, model in enumerate(members, start=1):
        path = sources / f"{number}.ckpt"
        torch.save(model.state_dict(), path)
        records.append(
            {
                "member": number,
                "path": f"ckpt_files/fine-tuned/{number}.ckpt",
                "sha256": _sha256(path),
            }
        )
    contract = tmp_path / "sources.json"
    contract.write_text(
        json.dumps(
            {
                "format_version": 1,
                "source_bundle_version": "1.0.0",
                "source_commit": "0" * 40,
                "members": records,
            }
        ),
        encoding="utf-8",
    )
    return sources, contract


def rewrite_contract_hash(contract, changed_path):
    payload = json.loads(contract.read_text(encoding="utf-8"))
    number = int(changed_path.stem)
    payload["members"][number - 1]["sha256"] = _sha256(changed_path)
    contract.write_text(json.dumps(payload), encoding="utf-8")


def _expected_bundle_state(sources):
    expected = {}
    for number in range(1, 11):
        state = torch.load(sources / f"{number}.ckpt", weights_only=True)
        if number == 1:
            for key, value in state.items():
                if not key.startswith("model_final."):
                    expected[f"trunk.{key}"] = value
        for key, value in state.items():
            if key.startswith("model_final."):
                expected[f"heads.{number - 1}.{key.removeprefix('model_final.')}"] = value
    return expected


def _raw_safetensors_header(path):
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        return json.loads(stream.read(header_size))


def test_repository_source_contract_is_exact_and_immutable():
    contract_path = Path(converter.__file__).with_name("gamma_ensemble_sources.json")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))

    assert payload == EXPECTED_SOURCE_CONTRACT
    assert list(payload) == [
        "format_version",
        "source_bundle_version",
        "source_commit",
        "members",
    ]
    assert [entry["member"] for entry in payload["members"]] == list(range(1, 11))
    assert all(list(entry) == ["member", "path", "sha256"] for entry in payload["members"])
    assert all(entry["sha256"] == entry["sha256"].lower() for entry in payload["members"])


def test_converter_builds_exact_93_tensor_bundle(tmp_path):
    sources, contract = write_legacy_sources(tmp_path)
    output = tmp_path / "bundle.safetensors"
    expected = _expected_bundle_state(sources)

    report = converter.build_bundle(sources, output, contract)

    assert report.path == output
    assert report.sha256 == _sha256(output)
    assert report.tensor_count == 93
    assert report.tensor_bytes == 5_202_560
    with pytest.raises(FrozenInstanceError):
        report.tensor_count = 0
    actual = load_file(output, device="cpu")
    assert set(actual) == set(GammaEnsemble(member_count=10).state_dict())
    for key in expected:
        assert torch.equal(actual[key], expected[key])
    header = _raw_safetensors_header(output)
    assert list(header) == ["__metadata__", *sorted(expected)]
    with safe_open(output, framework="pt", device="cpu") as handle:
        assert handle.metadata() == {
            "asset_name": "gamma-tuned-ensemble",
            "format_version": "1",
            "member_count": "10",
            "shared_tensor_count": "33",
            "head_tensor_count": "60",
            "source_manifest_bundle_version": "1.0.0",
        }


def test_converter_rejects_a_shared_tensor_that_differs(tmp_path):
    sources, contract = write_legacy_sources(tmp_path)
    changed = torch.load(sources / "2.ckpt", weights_only=True)
    changed["model_sigma.0.weight"] = changed["model_sigma.0.weight"].clone()
    changed["model_sigma.0.weight"][0, 0] += 1
    torch.save(changed, sources / "2.ckpt")
    rewrite_contract_hash(contract, sources / "2.ckpt")

    with pytest.raises(ValueError, match="shared tensor differs.*model_sigma.0.weight"):
        converter.build_bundle(sources, tmp_path / "bundle.safetensors", contract)


def test_converter_rejects_an_unexpected_checkpoint_key(tmp_path):
    sources, contract = write_legacy_sources(tmp_path)
    changed = torch.load(sources / "4.ckpt", weights_only=True)
    changed["unexpected.weight"] = torch.ones(1)
    torch.save(changed, sources / "4.ckpt")
    rewrite_contract_hash(contract, sources / "4.ckpt")

    with pytest.raises(ValueError, match="state keys mismatch.*unexpected.weight"):
        converter.build_bundle(sources, tmp_path / "bundle.safetensors", contract)


def test_converter_rejects_a_missing_member(tmp_path):
    sources, contract = write_legacy_sources(tmp_path)
    (sources / "10.ckpt").unlink()

    with pytest.raises(FileNotFoundError, match="ckpt_files/fine-tuned/10.ckpt"):
        converter.build_bundle(sources, tmp_path / "bundle.safetensors", contract)


def test_converter_rejects_an_extra_numbered_checkpoint(tmp_path):
    sources, contract = write_legacy_sources(tmp_path)
    (sources / "11.ckpt").write_bytes(b"not approved")

    with pytest.raises(ValueError, match="unexpected checkpoint files.*11.ckpt"):
        converter.build_bundle(sources, tmp_path / "bundle.safetensors", contract)


def test_converter_rejects_a_wrong_hash_before_loading(tmp_path, monkeypatch):
    sources, contract = write_legacy_sources(tmp_path)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["members"][0]["sha256"] = "f" * 64
    contract.write_text(json.dumps(payload), encoding="utf-8")

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("torch.load must not run before digest verification")

    monkeypatch.setattr(converter.torch, "load", fail_if_loaded)
    with pytest.raises(ValueError, match="SHA256 mismatch.*ckpt_files/fine-tuned/1.ckpt"):
        converter.build_bundle(sources, tmp_path / "bundle.safetensors", contract)


@pytest.mark.parametrize("corruption", ["dtype", "shape"])
def test_converter_rejects_a_wrong_tensor_dtype_or_shape(tmp_path, corruption):
    sources, contract = write_legacy_sources(tmp_path)
    changed = torch.load(sources / "3.ckpt", weights_only=True)
    key = "model_final.0.weight"
    if corruption == "dtype":
        changed[key] = changed[key].to(torch.float64)
    else:
        changed[key] = changed[key][:-1].clone()
    torch.save(changed, sources / "3.ckpt")
    rewrite_contract_hash(contract, sources / "3.ckpt")

    with pytest.raises(ValueError, match=f"tensor {corruption} mismatch.*{key}"):
        converter.build_bundle(sources, tmp_path / "bundle.safetensors", contract)


def test_converter_rejects_a_symlink_source(tmp_path):
    sources, contract = write_legacy_sources(tmp_path)
    original = sources / "6.ckpt"
    target = tmp_path / "member-six.ckpt"
    original.replace(target)
    original.symlink_to(target)

    with pytest.raises(ValueError, match="symlink.*ckpt_files/fine-tuned/6.ckpt"):
        converter.build_bundle(sources, tmp_path / "bundle.safetensors", contract)


def test_converter_preserves_an_existing_output_before_source_access(tmp_path):
    output = tmp_path / "bundle.safetensors"
    output.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="output path already exists"):
        converter.build_bundle(tmp_path / "missing", output, tmp_path / "missing.json")

    assert output.read_bytes() == b"keep me"


def test_converter_removes_only_its_temporary_file_after_interrupted_save(
    tmp_path, monkeypatch
):
    sources, contract = write_legacy_sources(tmp_path)
    source_bytes = {path.name: path.read_bytes() for path in sources.iterdir()}
    output = tmp_path / "bundle.safetensors"
    unrelated = tmp_path / ".unrelated.tmp"
    unrelated.write_bytes(b"keep")

    def interrupted_save(state, path, metadata):
        Path(path).write_bytes(b"partial bundle")
        raise OSError("interrupted save")

    monkeypatch.setattr(converter, "save_file", interrupted_save)
    with pytest.raises(OSError, match="interrupted save"):
        converter.build_bundle(sources, output, contract)

    assert not output.exists()
    assert unrelated.read_bytes() == b"keep"
    assert {path.name: path.read_bytes() for path in sources.iterdir()} == source_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        ".unrelated.tmp",
        "sources",
        "sources.json",
    ]


def test_converter_strictly_reloads_before_installing_output(tmp_path, monkeypatch):
    sources, contract = write_legacy_sources(tmp_path)
    real_load_file = converter.load_file
    calls = []

    def corrupt_reload(path, device):
        calls.append((Path(path), device))
        state = real_load_file(path, device=device)
        state["trunk.model_sigma.0.weight"] = torch.zeros_like(
            state["trunk.model_sigma.0.weight"]
        )
        return state

    monkeypatch.setattr(converter, "load_file", corrupt_reload)
    output = tmp_path / "bundle.safetensors"
    with pytest.raises(ValueError, match="post-save tensor differs.*trunk.model_sigma.0.weight"):
        converter.build_bundle(sources, output, contract)

    assert len(calls) == 1
    assert calls[0][0].parent == output.parent
    assert calls[0][0] != output
    assert calls[0][1] == "cpu"
    assert not output.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["sources", "sources.json"]


def test_independent_conversions_are_byte_identical(tmp_path):
    sources, contract = write_legacy_sources(tmp_path)
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    converter.build_bundle(sources, first, contract)
    converter.build_bundle(sources, second, contract)

    assert first.read_bytes() == second.read_bytes()


def test_cli_accepts_positional_paths_and_prints_stable_report_fields(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "bundle.safetensors"
    report = converter.BundleReport(
        path=output,
        sha256="a" * 64,
        tensor_count=93,
        tensor_bytes=5_202_560,
    )
    calls = []
    monkeypatch.setattr(
        converter,
        "build_bundle",
        lambda *args: calls.append(args) or report,
    )

    assert converter.main(
        [
            str(tmp_path / "sources"),
            str(output),
            "--source-contract",
            str(tmp_path / "sources.json"),
        ]
    ) == 0
    assert calls == [(tmp_path / "sources", output, tmp_path / "sources.json")]
    assert capsys.readouterr().out.splitlines() == [
        f"path={output}",
        f"sha256={'a' * 64}",
        "tensor_count=93",
        "tensor_bytes=5202560",
    ]
