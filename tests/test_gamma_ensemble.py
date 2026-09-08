import pytest
import torch

from tennetsac.models.GammaEnsemble import GammaEnsemble
from tennetsac.models.Prf2Gamma import Prf_to_Seg_Model


SHARED_PREFIXES = (
    "model_sigma.",
    "bn_sig.",
    "temp_embedding.",
    "bn_t.",
    "model_combined.",
    "bn2.",
    "res_block.",
)

LEGACY_STATE_KEYS = (
    "model_sigma.0.weight",
    "model_sigma.0.bias",
    "model_sigma.2.weight",
    "model_sigma.2.bias",
    "model_sigma.4.weight",
    "model_sigma.4.bias",
    "bn_sig.weight",
    "bn_sig.bias",
    "bn_sig.running_mean",
    "bn_sig.running_var",
    "bn_sig.num_batches_tracked",
    "temp_embedding.0.weight",
    "temp_embedding.0.bias",
    "temp_embedding.2.weight",
    "temp_embedding.2.bias",
    "bn_t.weight",
    "bn_t.bias",
    "bn_t.running_mean",
    "bn_t.running_var",
    "bn_t.num_batches_tracked",
    "model_combined.0.weight",
    "model_combined.0.bias",
    "model_combined.2.weight",
    "model_combined.2.bias",
    "bn2.weight",
    "bn2.bias",
    "bn2.running_mean",
    "bn2.running_var",
    "bn2.num_batches_tracked",
    "res_block.fc1.weight",
    "res_block.fc1.bias",
    "res_block.fc2.weight",
    "res_block.fc2.bias",
    "model_final.0.weight",
    "model_final.0.bias",
    "model_final.2.weight",
    "model_final.2.bias",
    "model_final.4.weight",
    "model_final.4.bias",
)


def _legacy_members(count=3):
    torch.manual_seed(4107)
    first = Prf_to_Seg_Model().eval()
    members = [first]
    for seed in range(4108, 4108 + count - 1):
        torch.manual_seed(seed)
        member = Prf_to_Seg_Model().eval()
        state = member.state_dict()
        for key, value in first.state_dict().items():
            if key.startswith(SHARED_PREFIXES):
                state[key].copy_(value)
        members.append(member)
    return members


def _ensemble_from_legacy(members):
    packed = {}
    for key, value in members[0].state_dict().items():
        if not key.startswith("model_final."):
            packed[f"trunk.{key}"] = value
    for index, member in enumerate(members):
        for key, value in member.state_dict().items():
            if key.startswith("model_final."):
                suffix = key.removeprefix("model_final.")
                packed[f"heads.{index}.{suffix}"] = value
    ensemble = GammaEnsemble(member_count=len(members))
    ensemble.load_state_dict(packed, strict=True)
    return ensemble


def test_mean_member_and_all_member_gradients_match_legacy_models():
    legacy = _legacy_members()
    ensemble = _ensemble_from_legacy(legacy).eval()
    sigma = torch.linspace(0.1, 1.1, 102).reshape(2, 51)
    temperature = torch.tensor([298.15, 315.0])
    expected = torch.stack(
        [model(sigma.clone(), temperature)[1] for model in legacy]
    )

    torch.testing.assert_close(
        ensemble.predict_segac(sigma, temperature, return_members=True),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        ensemble.predict_segac(sigma, temperature),
        expected.mean(dim=0),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        ensemble.predict_segac(sigma, temperature, member_index=1),
        expected[1],
        rtol=1e-5,
        atol=1e-6,
    )


def test_member_prediction_expands_scalar_temperature_to_sigma_batch_size():
    ensemble = _ensemble_from_legacy(_legacy_members()).eval()
    sigma = torch.linspace(0.1, 1.1, 102).reshape(2, 51)

    scalar_result = ensemble.predict_segac(sigma, 298.15, member_index=0)
    batch_result = ensemble.predict_segac(
        sigma, torch.full((2,), 298.15), member_index=0
    )

    torch.testing.assert_close(scalar_result, batch_result, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("temperature", [torch.ones(3), torch.ones(2, 2)])
def test_temperature_rejects_a_non_scalar_value_that_does_not_match_batch(
    temperature,
):
    ensemble = GammaEnsemble(member_count=3).eval()
    sigma = torch.ones(2, 51)

    with pytest.raises(
        ValueError, match="temperature must be scalar or match sigma batch size"
    ):
        ensemble.predict_segac(sigma, temperature, member_index=0)


@pytest.mark.parametrize("member_count", [0, -1, True, 1.5])
def test_member_count_must_be_a_positive_integer(member_count):
    with pytest.raises(ValueError, match="member_count must be a positive integer"):
        GammaEnsemble(member_count=member_count)


@pytest.mark.parametrize("member_index", [-1, 3, True, 1.0])
def test_member_index_rejects_invalid_values(member_index):
    ensemble = GammaEnsemble(member_count=3).eval()

    with pytest.raises(IndexError, match="member_index out of range"):
        ensemble.predict_segac(torch.ones(2, 51), 298.15, member_index=member_index)


def test_member_index_and_return_members_are_mutually_exclusive():
    ensemble = GammaEnsemble(member_count=3).eval()

    with pytest.raises(
        ValueError, match="member_index and return_members are mutually exclusive"
    ):
        ensemble.predict_segac(
            torch.ones(2, 51), 298.15, member_index=0, return_members=True
        )


def test_mean_member_and_all_member_modes_return_exact_shapes():
    ensemble = GammaEnsemble(member_count=3).eval()
    sigma = torch.linspace(0.1, 1.1, 102).reshape(2, 51)

    assert ensemble.predict_segac(sigma, 298.15).shape == (2, 51)
    assert ensemble.predict_segac(sigma, 298.15, member_index=0).shape == (2, 51)
    assert ensemble.predict_segac(sigma, 298.15, return_members=True).shape == (
        3,
        2,
        51,
    )


def test_prediction_rejects_training_mode():
    ensemble = GammaEnsemble(member_count=3)

    with pytest.raises(RuntimeError, match="GammaEnsemble prediction requires eval mode"):
        ensemble.predict_segac(torch.ones(2, 51), 298.15)


@pytest.mark.parametrize(
    "options",
    [{}, {"member_index": 0}, {"return_members": True}],
    ids=["mean", "member", "all-members"],
)
def test_mean_member_and_all_member_prediction_modes_call_the_trunk_once(options):
    ensemble = GammaEnsemble(member_count=3).eval()
    calls = []
    handle = ensemble.trunk.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(None)
    )

    try:
        ensemble.predict_segac(torch.ones(2, 51), 298.15, **options)
    finally:
        handle.remove()

    assert len(calls) == 1


def test_batched_vjp_matches_ten_explicit_legacy_gradients():
    legacy = _legacy_members(count=10)
    ensemble = _ensemble_from_legacy(legacy).eval()
    sigma = torch.linspace(0.1, 1.1, 102).reshape(2, 51)
    temperature = torch.tensor([298.15, 315.0])
    expected = torch.stack(
        [model(sigma.clone(), temperature)[1] for model in legacy]
    )

    actual = ensemble.predict_segac(sigma, temperature, return_members=True)

    assert actual.shape == (10, 2, 51)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_legacy_model_keeps_exact_state_dict_names():
    assert tuple(Prf_to_Seg_Model().state_dict()) == LEGACY_STATE_KEYS


def test_ensemble_segac_delegates_to_the_ensemble_mean_and_returns_cpu_tensor():
    from tennetsac.utils.property import ensemble_segac

    calls = []

    class FakeEnsemble:
        def predict_segac(self, sigma, temperature):
            calls.append((sigma, temperature))
            return torch.full_like(sigma, 3.25)

    sigma = torch.ones(1, 51)
    result = ensemble_segac(FakeEnsemble(), sigma, 298.15)

    assert calls == [(sigma, 298.15)]
    assert result.device.type == "cpu"
    torch.testing.assert_close(result, torch.full_like(sigma, 3.25))


def test_core_ensemble_predictor_forwards_modes_to_one_runtime_ensemble(monkeypatch):
    from tennetsac import core

    calls = []

    class FakeEnsemble:
        def predict_segac(
            self, sigma, temperature, *, member_index=None, return_members=False
        ):
            calls.append((sigma, temperature, member_index, return_members))
            return torch.full_like(sigma, 2.5)

    monkeypatch.setattr(
        core, "get_runtime", lambda: type("Runtime", (), {"gamma_ensemble": FakeEnsemble()})()
    )
    sigma = torch.ones(1, 51)

    result = core.ensemble_predictor(
        sigma, 298.15, member_index=2, return_members=False
    )

    assert calls == [(sigma, 298.15, 2, False)]
    assert result.device.type == "cpu"
    torch.testing.assert_close(result, torch.full_like(sigma, 2.5))
