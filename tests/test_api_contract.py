import inspect

import numpy as np
import pytest
import torch


EXPECTED = {
    "profile": "(smiles: str) -> Tuple[List[float], float, float]",
    "binary_lng": "(smiles: List[str], temperature: float, molefraction: List[float], version: str = 'tuned') -> Tuple[List[float], List[float]]",
    "multi_lng": "(smiles: List[str], temperature: float, composition: List[float], version: str = 'tuned') -> List[float]",
    "fit_nrtl": "(smiles1, smiles2, alpha=0.3, temp_range=None, x_points=21)",
    "plot_nrtl_fitting": "(smiles1, smiles2, fit_result)",
}


def test_prediction_signatures_are_stable():
    import tennetsac

    for name, expected in EXPECTED.items():
        assert str(inspect.signature(getattr(tennetsac, name))) == expected


def test_profile_returns_python_list_and_scalar_values(monkeypatch):
    import tennetsac
    from tennetsac import core

    monkeypatch.setattr(
        core,
        "sigma_profile_wrapper",
        lambda smiles: (torch.tensor([[1.0, 2.0]]), 3.5, 4.5),
    )

    result = tennetsac.profile("CCO")

    assert result == ([1.0, 2.0], 3.5, 4.5)
    assert isinstance(result[0], list)
    assert all(isinstance(value, float) for value in result[0])
    assert isinstance(result[1], float)
    assert isinstance(result[2], float)


def test_binary_lng_returns_python_lists(monkeypatch):
    import tennetsac
    from tennetsac import core

    monkeypatch.setattr(
        core,
        "calc_ln_gamma_binary",
        lambda *args, **kwargs: (np.array([0.1, 0.2]), np.array([0.3, 0.4])),
    )

    result = tennetsac.binary_lng(["CCO", "O"], 298.15, [0.25, 0.75])

    assert result == ([0.1, 0.2], [0.3, 0.4])
    assert all(isinstance(values, list) for values in result)


def test_multi_lng_returns_python_list(monkeypatch):
    import tennetsac
    from tennetsac import core

    monkeypatch.setattr(
        core,
        "calc_ln_gamma",
        lambda *args, **kwargs: np.array([0.1, 0.2, 0.3]),
    )

    result = tennetsac.multi_lng(["CCO", "O", "N"], 298.15, [0.2, 0.3, 0.5])

    assert result == [0.1, 0.2, 0.3]
    assert isinstance(result, list)


def test_multi_lng_autocompletes_an_n_minus_one_composition(monkeypatch):
    import tennetsac
    from tennetsac import core

    profiles = {
        "A": (torch.tensor([[1.0, 0.5]]), 2.0, 3.0),
        "B": (torch.tensor([[0.5, 1.0]]), 3.0, 4.0),
        "C": (torch.tensor([[1.5, 1.0]]), 4.0, 5.0),
    }
    monkeypatch.setattr(core, "sigma_profile_wrapper", profiles.__getitem__)
    monkeypatch.setattr(
        core,
        "select_gamma_predictor",
        lambda version: lambda sigma, temperature: torch.zeros_like(sigma),
    )

    completed = tennetsac.multi_lng(["A", "B", "C"], 298.15, [0.3, 0.4])
    explicit = tennetsac.multi_lng(["A", "B", "C"], 298.15, [0.3, 0.4, 0.3])

    assert completed == pytest.approx(explicit)
    assert len(completed) == 3


def test_binary_lng_rejects_non_binary_smiles():
    import tennetsac

    with pytest.raises(
        ValueError,
        match=r"^'smiles' must be a list of exactly two SMILES strings, got",
    ):
        tennetsac.binary_lng(["CCO"], 298.15, [1.0])


def test_invalid_model_version_raises_existing_error():
    import tennetsac

    with pytest.raises(
        ValueError,
        match=r"^Invalid model_type\. Choose 'base' or 'tuned'\.$",
    ):
        tennetsac.binary_lng(["CCO", "O"], 298.15, [0.5, 0.5], version="invalid")
