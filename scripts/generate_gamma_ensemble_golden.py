#!/usr/bin/env python3
"""Generate the independently reproducible legacy gamma parity fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from build_gamma_ensemble_asset import _load_source_contract, _load_verified_members
from tennetsac.models.Prf2Gamma import Prf_to_Seg_Model
from tennetsac.utils.property import calc_ln_gamma, calc_ln_gamma_binary


FIXTURE_COMMAND = (
    "conda run -n tsac_env python scripts/generate_gamma_ensemble_golden.py "
    "--legacy-dir /private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908 "
    "--source-contract scripts/gamma_ensemble_sources.json "
    "--output tests/fixtures/gamma_ensemble_golden.json"
)


def _positive_profile(start: float, stop: float) -> torch.Tensor:
    return torch.linspace(start, stop, 102, dtype=torch.float32).reshape(2, 51)


GRADIENT_CASES = (
    ("profile_a_298_15_k", _positive_profile(0.011, 0.521), 298.15),
    ("profile_b_343_15_k", _positive_profile(0.023, 0.733), 343.15),
)
COMPONENTS = (
    {"name": "component-1", "sigma": _positive_profile(0.017, 0.417), "area": 91.25, "volume": 112.5},
    {"name": "component-2", "sigma": _positive_profile(0.029, 0.609), "area": 105.75, "volume": 138.25},
    {"name": "component-3", "sigma": _positive_profile(0.041, 0.801), "area": 119.5, "volume": 164.0},
)


def _as_json(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def _legacy_models(legacy_dir: Path, source_contract: Path) -> list[Prf_to_Seg_Model]:
    contract = _load_source_contract(source_contract)
    states = _load_verified_members(legacy_dir, contract)
    models = []
    for state in states:
        model = Prf_to_Seg_Model().eval()
        model.load_state_dict(state, strict=True)
        models.append(model)
    return models


def _profile_lookup(components: tuple[dict, ...]):
    values = {
        component["name"]: (
            component["sigma"],
            component["area"],
            component["volume"],
        )
        for component in components
    }
    return lambda name: values[name]


def _predict(model: Prf_to_Seg_Model, sigma: torch.Tensor, temperature: float) -> torch.Tensor:
    temperatures = torch.full(
        (sigma.shape[0],), temperature, dtype=sigma.dtype, device=sigma.device
    )
    return model(sigma.clone(), temperatures)[1]


def _gradient_fixture(models: list[Prf_to_Seg_Model]) -> list[dict]:
    cases = []
    for name, sigma, temperature in GRADIENT_CASES:
        members = torch.stack([_predict(model, sigma, temperature) for model in models])
        cases.append(
            {
                "name": name,
                "sigma": _as_json(sigma),
                "temperature": temperature,
                "members": _as_json(members),
                "mean": _as_json(members.mean(dim=0)),
                "std": _as_json(members.std(dim=0, unbiased=False)),
            }
        )
    return cases


def _mixture_fixture(models: list[Prf_to_Seg_Model]) -> list[dict]:
    serial_components = [
        {**component, "sigma": _as_json(component["sigma"])} for component in COMPONENTS
    ]
    lookup = _profile_lookup(COMPONENTS)
    cases = []
    binary_composition = [0.2, 0.6]
    binary_members = []
    for model in models:
        predictor = lambda sigma, temperature, model=model: _predict(model, sigma, temperature)
        left, right = calc_ln_gamma_binary(
            "component-1", "component-2", binary_composition, 298.15, predictor, lookup
        )
        binary_members.append([left.tolist(), right.tolist()])
    binary_tensor = torch.tensor(binary_members, dtype=torch.float64)
    cases.append(
        {
            "name": "binary_298_15_k",
            "kind": "binary",
            "temperature": 298.15,
            "composition": binary_composition,
            "components": serial_components[:2],
            "members": binary_members,
            "mean": _as_json(binary_tensor.mean(dim=0)),
            "std": _as_json(binary_tensor.std(dim=0, unbiased=False)),
        }
    )
    ternary_composition = [0.2, 0.3, 0.5]
    ternary_members = []
    for model in models:
        predictor = lambda sigma, temperature, model=model: _predict(model, sigma, temperature)
        ternary_members.append(
            calc_ln_gamma(
                [component["name"] for component in COMPONENTS],
                ternary_composition,
                343.15,
                predictor,
                lookup,
            ).tolist()
        )
    ternary_tensor = torch.tensor(ternary_members, dtype=torch.float64)
    cases.append(
        {
            "name": "ternary_343_15_k",
            "kind": "ternary",
            "temperature": 343.15,
            "composition": ternary_composition,
            "components": serial_components,
            "members": ternary_members,
            "mean": _as_json(ternary_tensor.mean(dim=0)),
            "std": _as_json(ternary_tensor.std(dim=0, unbiased=False)),
        }
    )
    return cases


def generate(legacy_dir: Path, source_contract: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"golden fixture already exists: {output}")
    contract = _load_source_contract(source_contract)
    models = _legacy_models(legacy_dir, source_contract)
    payload = {
        "ddof": 0,
        "generator_command": FIXTURE_COMMAND,
        "gradient_cases": _gradient_fixture(models),
        "mixture_cases": _mixture_fixture(models),
        "pytorch_version": torch.__version__,
        "source_commit": contract["source_commit"],
        "source_digests": [
            {"member": entry["member"], "path": entry["path"], "sha256": entry["sha256"]}
            for entry in contract["members"]
        ],
    }
    output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    generate(args.legacy_dir, args.source_contract, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
