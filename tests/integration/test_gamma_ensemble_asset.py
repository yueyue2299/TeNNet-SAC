"""Real parity checks for a migrated gamma-ensemble candidate."""

import hashlib
import json
import os
from pathlib import Path

import pytest

EXPECTED_CANDIDATE_SHA256 = "9ebd1b3b72c406c1987afd7ee842cdbc152c019148e6c97aea5e39cf74a79e22"

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_candidate_matches_the_independent_legacy_golden_fixture():
    legacy_value = os.environ.get("TENNETSAC_LEGACY_GAMMA_DIR")
    candidate_value = os.environ.get("TENNETSAC_GAMMA_ENSEMBLE")
    if not legacy_value or not candidate_value:
        pytest.skip("set TENNETSAC_LEGACY_GAMMA_DIR and TENNETSAC_GAMMA_ENSEMBLE")

    # Imports that initialize PyTorch occur only after the environment skip.
    import torch

    from tennetsac.gamma_ensemble import BUNDLE_TENSOR_BYTES, BUNDLE_TENSOR_COUNT
    from tennetsac.gamma_ensemble import load_gamma_ensemble
    from tennetsac.models.Prf2Gamma import Prf_to_Seg_Model
    from tennetsac.utils.property import calc_ln_gamma, calc_ln_gamma_binary

    root = Path(__file__).parents[2]
    fixture = json.loads(
        (root / "tests/fixtures/gamma_ensemble_golden.json").read_text("utf-8")
    )
    legacy_dir = Path(legacy_value)
    candidate = Path(candidate_value)
    contract = json.loads((root / "scripts/gamma_ensemble_sources.json").read_text("utf-8"))
    actual_sources = [
        {
            "member": entry["member"],
            "path": entry["path"],
            "sha256": _sha256(legacy_dir / Path(entry["path"]).name),
        }
        for entry in contract["members"]
    ]
    assert actual_sources == fixture["source_digests"]
    assert _sha256(candidate) == EXPECTED_CANDIDATE_SHA256

    ensemble = load_gamma_ensemble(candidate, EXPECTED_CANDIDATE_SHA256)
    assert len(ensemble.state_dict()) == BUNDLE_TENSOR_COUNT
    assert sum(
        value.numel() * value.element_size()
        for value in ensemble.state_dict().values()
    ) == BUNDLE_TENSOR_BYTES
    legacy = []
    for number in range(1, 11):
        model = Prf_to_Seg_Model().eval()
        model.load_state_dict(
            torch.load(legacy_dir / f"{number}.ckpt", map_location="cpu", weights_only=True),
            strict=True,
        )
        legacy.append(model)

    for case in fixture["gradient_cases"]:
        sigma = torch.tensor(case["sigma"], dtype=torch.float32)
        expected_members = torch.tensor(case["members"], dtype=torch.float32)
        expected_mean = torch.tensor(case["mean"], dtype=torch.float32)
        expected_std = torch.tensor(case["std"], dtype=torch.float32)
        actual_members = ensemble.predict_segac(
            sigma, case["temperature"], return_members=True
        )
        temperatures = torch.full((sigma.shape[0],), case["temperature"])
        direct_members = torch.stack(
            [model(sigma.clone(), temperatures)[1] for model in legacy]
        )
        torch.testing.assert_close(actual_members, expected_members, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_members, direct_members, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_members.mean(dim=0), expected_mean, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            actual_members.std(dim=0, unbiased=False), expected_std, rtol=1e-5, atol=1e-6
        )

    def profile_lookup(components):
        profiles = {
            component["name"]: (
                torch.tensor(component["sigma"], dtype=torch.float32),
                component["area"],
                component["volume"],
            )
            for component in components
        }
        return lambda name: profiles[name]

    for case in fixture["mixture_cases"]:
        lookup = profile_lookup(case["components"])
        member_values = []
        for model in legacy:
            predictor = lambda sigma, temperature, model=model: model(
                sigma,
                torch.full((sigma.shape[0],), temperature, dtype=sigma.dtype),
            )[1]
            if case["kind"] == "binary":
                left, right = calc_ln_gamma_binary(
                    "component-1", "component-2", case["composition"], case["temperature"], predictor, lookup
                )
                member_values.append([left.tolist(), right.tolist()])
            else:
                member_values.append(
                    calc_ln_gamma(
                        [item["name"] for item in case["components"]],
                        case["composition"], case["temperature"], predictor, lookup
                    ).tolist()
                )
        actual = torch.tensor(member_values, dtype=torch.float64)
        torch.testing.assert_close(
            actual, torch.tensor(case["members"], dtype=torch.float64), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            actual.mean(dim=0), torch.tensor(case["mean"], dtype=torch.float64), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            actual.std(dim=0, unbiased=False),
            torch.tensor(case["std"], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-6,
        )
