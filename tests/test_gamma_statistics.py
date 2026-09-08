import numpy as np
import torch

from tennetsac.utils.property import calc_ln_gamma, calc_ln_gamma_binary


MEMBER_VALUES = {
    0.2: [0.0, 8.0, 2.0],
    0.65: [7.0, 3.0, 6.0],
    0.8: [4.0, 1.0, 9.0],
    0.88: [2.0, 12.0, 9.0],
    1.2: [5.0, 11.0, 3.0],
}
PROFILES = {
    "A": (torch.full((1, 51), 0.2), 2.0, 3.0),
    "B": (torch.full((1, 51), 0.8), 3.0, 4.0),
    "C": (torch.full((1, 51), 1.2), 4.0, 5.0),
}


def _fixed_member_predictor(sigma, _temperature):
    value = round(float(sigma.mean()), 2)
    member_values = torch.tensor(MEMBER_VALUES[value], dtype=sigma.dtype)
    return member_values[:, None, None].expand(-1, sigma.shape[0], sigma.shape[1])


def _manual_binary_member_values(x1_list):
    aeff = 5.8447
    sigma_a, area_a, volume_a = PROFILES["A"]
    sigma_b, area_b, volume_b = PROFILES["B"]
    pure_a = np.asarray(MEMBER_VALUES[0.2])
    pure_b = np.asarray(MEMBER_VALUES[0.8])
    values = []
    for x1 in x1_list:
        x2 = 1.0 - x1
        mix = np.asarray(MEMBER_VALUES[round(0.2 * x1 + 0.8 * x2, 2)])
        total_area = x1 * area_a + x2 * area_b
        total_volume = x1 * volume_a + x2 * volume_b
        reference_area = 79.531954
        coordination = 5.0
        combinatorial = []
        for area, volume in ((area_a, volume_a), (area_b, volume_b)):
            r = volume / total_volume
            q = area / total_area
            combinatorial.append(
                1
                - r
                + np.log(r)
                - coordination * (area / reference_area) * (1 - r / q + np.log(r / q))
            )
        residual_a = area_a / aeff * np.sum((sigma_a.numpy() / area_a)) * (mix - pure_a)
        residual_b = area_b / aeff * np.sum((sigma_b.numpy() / area_b)) * (mix - pure_b)
        values.append(np.stack((combinatorial[0] + residual_a, combinatorial[1] + residual_b), axis=-1))
    return np.stack(values, axis=1)


def _manual_multi_member_values():
    aeff = 5.8447
    composition = [0.2, 0.3, 0.5]
    pure_values = [
        np.asarray(MEMBER_VALUES[0.2]),
        np.asarray(MEMBER_VALUES[0.8]),
        np.asarray(MEMBER_VALUES[1.2]),
    ]
    mix = np.asarray(MEMBER_VALUES[0.88])
    total_area = sum(x * PROFILES[name][1] for name, x in zip(("A", "B", "C"), composition))
    total_volume = sum(x * PROFILES[name][2] for name, x in zip(("A", "B", "C"), composition))
    reference_area = 79.531954
    coordination = 5.0
    per_component = []
    for name, pure in zip(("A", "B", "C"), pure_values):
        sigma, area, volume = PROFILES[name]
        r = volume / total_volume
        q = area / total_area
        combinatorial = (
            1
            - r
            + np.log(r)
            - coordination * (area / reference_area) * (1 - r / q + np.log(r / q))
        )
        residual = area / aeff * np.sum(sigma.numpy() / area) * (mix - pure)
        per_component.append(combinatorial + residual)
    return np.stack(per_component, axis=-1)


def test_binary_statistics_aggregate_complete_paired_member_lng_values():
    x1 = [0.25]
    member_values = _manual_binary_member_values(x1)

    mean_1, mean_2, std_1, std_2 = calc_ln_gamma_binary(
        "A", "B", x1, 298.15, _fixed_member_predictor, PROFILES.__getitem__, return_std=True
    )

    np.testing.assert_allclose(mean_1, member_values[:, :, 0].mean(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(mean_2, member_values[:, :, 1].mean(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(std_1, member_values[:, :, 0].std(axis=0, ddof=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(std_2, member_values[:, :, 1].std(axis=0, ddof=0), rtol=1e-5, atol=1e-6)

    separately_aggregated_std = np.std(MEMBER_VALUES[0.65], ddof=0) + np.std(MEMBER_VALUES[0.2], ddof=0)
    assert not np.isclose(std_1[0], separately_aggregated_std)


def test_multi_statistics_aggregate_complete_paired_member_lng_values():
    member_values = _manual_multi_member_values()

    mean, std = calc_ln_gamma(
        ["A", "B", "C"], [0.2, 0.3, 0.5], 298.15,
        _fixed_member_predictor, PROFILES.__getitem__, return_std=True,
    )

    np.testing.assert_allclose(mean, member_values.mean(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(std, member_values.std(axis=0, ddof=0), rtol=1e-5, atol=1e-6)

    separately_aggregated_std = np.std(MEMBER_VALUES[0.88], ddof=0) + np.std(MEMBER_VALUES[0.2], ddof=0)
    assert not np.isclose(std[0], separately_aggregated_std)
