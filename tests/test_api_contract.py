import inspect

import numpy as np
import pytest
import torch


EXPECTED = {
    "profile": "(smiles: str) -> Tuple[List[float], float, float]",
    "binary_lng": "(smiles: List[str], temperature: float, molefraction: List[float], version: str = 'tuned', return_std: bool = True) -> Tuple[List[float], List[float], List[float], List[float]]",
    "multi_lng": "(smiles: List[str], temperature: float, composition: List[float], version: str = 'tuned', return_std: bool = True) -> Tuple[List[float], List[float]]",
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


def test_binary_lng_returns_statistics_by_default(monkeypatch):
    import tennetsac
    from tennetsac import core

    monkeypatch.setattr(
        core,
        "calc_ln_gamma_binary",
        lambda *args, **kwargs: tuple(
            np.array(values)
            for values in ([1, 2], [3, 4], [0.1, 0.2], [0.3, 0.4])
        ),
    )

    result = tennetsac.binary_lng(["CCO", "O"], 298.15, [0.25, 0.75])

    assert result == ([1, 2], [3, 4], [0.1, 0.2], [0.3, 0.4])
    assert all(isinstance(values, list) for values in result)


def test_binary_lng_return_std_false_preserves_two_list_shape(monkeypatch):
    import tennetsac
    from tennetsac import core

    monkeypatch.setattr(
        core,
        "calc_ln_gamma_binary",
        lambda *args, **kwargs: (np.array([0.1, 0.2]), np.array([0.3, 0.4])),
    )

    result = tennetsac.binary_lng(
        ["CCO", "O"], 298.15, [0.25, 0.75], return_std=False
    )

    assert result == ([0.1, 0.2], [0.3, 0.4])
    assert all(isinstance(values, list) for values in result)


def test_multi_lng_returns_statistics_by_default(monkeypatch):
    import tennetsac
    from tennetsac import core

    monkeypatch.setattr(
        core,
        "calc_ln_gamma",
        lambda *args, **kwargs: (np.array([0.1, 0.2, 0.3]), np.array([0.01, 0.02, 0.03])),
    )

    result = tennetsac.multi_lng(["CCO", "O", "N"], 298.15, [0.2, 0.3, 0.5])

    assert result == ([0.1, 0.2, 0.3], [0.01, 0.02, 0.03])
    assert all(isinstance(values, list) for values in result)


def test_multi_lng_return_std_false_preserves_one_list_shape(monkeypatch):
    import tennetsac
    from tennetsac import core

    monkeypatch.setattr(
        core,
        "calc_ln_gamma",
        lambda *args, **kwargs: np.array([0.1, 0.2, 0.3]),
    )

    result = tennetsac.multi_lng(
        ["CCO", "O", "N"], 298.15, [0.2, 0.3, 0.5], return_std=False
    )

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
        lambda version, *, return_std: lambda sigma, temperature: torch.zeros_like(sigma),
    )

    completed = tennetsac.multi_lng(
        ["A", "B", "C"], 298.15, [0.3, 0.4], return_std=False
    )
    explicit = tennetsac.multi_lng(
        ["A", "B", "C"], 298.15, [0.3, 0.4, 0.3], return_std=False
    )

    assert completed == pytest.approx(explicit)
    assert len(completed) == 3


def test_binary_lng_rejects_non_binary_smiles():
    import tennetsac

    with pytest.raises(
        ValueError,
        match=r"^'smiles' must be a list of exactly two SMILES strings, got",
    ):
        tennetsac.binary_lng(["CCO"], 298.15, [1.0])


@pytest.mark.parametrize("version", ["base", *map(str, range(1, 11))])
def test_single_model_versions_require_mean_only(version):
    import tennetsac

    with pytest.raises(
        ValueError,
        match="return_std=False",
    ):
        tennetsac.binary_lng(["CCO", "O"], 298.15, [0.5, 0.5], version=version)


@pytest.mark.parametrize("version, expected_member", [("1", 0), ("10", 9)])
def test_numbered_versions_select_the_matching_zero_based_ensemble_head(
    monkeypatch, version, expected_member
):
    import tennetsac
    from tennetsac import core

    calls = []

    def fake_ensemble_predictor(
        sigma, temperature, *, member_index=None, return_members=False
    ):
        calls.append((member_index, return_members))
        return torch.zeros_like(sigma)

    def calculate_with_selected_predictor(*args, **kwargs):
        predictor = kwargs["gamma_predictor"]
        predictor(torch.ones(1, 51), 298.15)
        return np.array([0.1]), np.array([0.2])

    monkeypatch.setattr(core, "ensemble_predictor", fake_ensemble_predictor)
    monkeypatch.setattr(core, "calc_ln_gamma_binary", calculate_with_selected_predictor)

    assert tennetsac.binary_lng(
        ["CCO", "O"], 298.15, [0.5], version=version, return_std=False
    ) == ([0.1], [0.2])
    assert calls == [(expected_member, False)]


@pytest.mark.parametrize("version", [1, 10, "0", "11", "invalid", None])
def test_invalid_versions_are_rejected_before_sigma_profile_work(monkeypatch, version):
    import tennetsac
    from tennetsac import core

    def unexpected_profile_work(_smiles):
        raise AssertionError("sigma-profile work must not run for an invalid version")

    monkeypatch.setattr(core, "sigma_profile_wrapper", unexpected_profile_work)

    with pytest.raises(
        ValueError,
        match=r"^version must be 'base', 'tuned', or a string from '1' to '10'$",
    ):
        tennetsac.binary_lng(["CCO", "O"], 298.15, [0.5], version=version)


def test_return_std_must_be_a_bool_before_sigma_profile_work(monkeypatch):
    import tennetsac
    from tennetsac import core

    monkeypatch.setattr(
        core,
        "sigma_profile_wrapper",
        lambda _smiles: (_ for _ in ()).throw(AssertionError("unexpected profile work")),
    )

    with pytest.raises(TypeError, match=r"^return_std must be a bool$"):
        tennetsac.multi_lng(["CCO", "O"], 298.15, [0.5, 0.5], return_std=1)
