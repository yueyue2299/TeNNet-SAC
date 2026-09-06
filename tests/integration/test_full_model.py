import json
import os
from pathlib import Path

import numpy as np
import pytest


SMILES = ["CCO", "ClCCCl"]
FIXTURE = Path(__file__).parents[1] / "fixtures" / "pypi_0_1_10_outputs.json"
CHEMBERTA_REPO = "DeepChem/ChemBERTa-77M-MLM"
CHEMBERTA_REVISION = "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"

pytestmark = pytest.mark.integration


def _require_local_models(monkeypatch):
    checkpoint = Path(os.environ.get("TENNETSAC_SMI_TED_CHECKPOINT", ""))
    if not checkpoint.is_file():
        pytest.skip("TENNETSAC_SMI_TED_CHECKPOINT does not name a local checkpoint")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    from huggingface_hub import try_to_load_from_cache
    from transformers import RobertaModel, RobertaTokenizer

    config = try_to_load_from_cache(
        CHEMBERTA_REPO, "config.json", revision=CHEMBERTA_REVISION
    )
    if not isinstance(config, str):
        pytest.skip("ChemBERTa2 is not cached at the pinned revision")
    try:
        RobertaTokenizer.from_pretrained(
            CHEMBERTA_REPO,
            revision=CHEMBERTA_REVISION,
            local_files_only=True,
        )
        RobertaModel.from_pretrained(
            CHEMBERTA_REPO,
            revision=CHEMBERTA_REVISION,
            local_files_only=True,
        )
    except OSError:
        pytest.skip("ChemBERTa2 cache is incomplete at the pinned revision")


def test_full_runtime_outputs_are_finite(monkeypatch):
    _require_local_models(monkeypatch)
    from tennetsac import binary_lng, profile

    sigma, area, volume = profile("CCO")
    lng1, lng2 = binary_lng(SMILES, 298.15, [0.25, 0.5, 0.75])

    assert len(sigma) == 51
    assert area > 0 and volume > 0
    assert np.isfinite([*sigma, *lng1, *lng2]).all()


def test_full_runtime_matches_pypi_0_1_10_golden_outputs(monkeypatch):
    _require_local_models(monkeypatch)
    from tennetsac import binary_lng, multi_lng, profile

    expected = json.loads(FIXTURE.read_text())
    sigma, area, volume = profile(expected["smiles"])
    lng1, lng2 = binary_lng(**{
        key: expected["binary"][key]
        for key in ("smiles", "temperature", "molefraction")
    })
    multi_lng_values = multi_lng(**{
        key: expected["multi"][key]
        for key in ("smiles", "temperature", "composition")
    })

    np.testing.assert_allclose(sigma, expected["sigma"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(area, expected["area"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(volume, expected["volume"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lng1, expected["binary"]["lng1"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lng2, expected["binary"]["lng2"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        multi_lng_values, expected["multi"]["lng"], rtol=1e-5, atol=1e-6
    )
