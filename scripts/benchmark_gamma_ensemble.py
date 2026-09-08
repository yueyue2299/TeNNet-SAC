#!/usr/bin/env python3
"""Measure the verified legacy gamma ensemble against its shared-trunk bundle.

The command intentionally benchmarks one narrow, fixed workload.  It is not a
general performance harness: it verifies the preserved legacy checkpoint set
and the packaged safetensors bundle before timing their mean and mean+standard
deviation paths.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

if __package__:
    from scripts.build_gamma_ensemble_asset import (
        _load_source_contract,
        _load_verified_members,
    )
else:
    from build_gamma_ensemble_asset import _load_source_contract, _load_verified_members

from tennetsac.gamma_ensemble import BUNDLE_TENSOR_BYTES, load_gamma_ensemble
from tennetsac.model_manifest import bundled_artifact
from tennetsac.models.Prf2Gamma import Prf_to_Seg_Model


DEFAULT_SOURCE_CONTRACT = Path(__file__).with_name("gamma_ensemble_sources.json")
FIXED_SIGMA_VALUE = 0.01
FIXED_SIGMA_SHAPE = (1, 51)
FIXED_TEMPERATURE_KELVIN = 298.15
REPEATED_BATCHES = 5
MIN_MEAN_SPEEDUP = 2.0
SPEED_THRESHOLD_EXIT = 2
MIN_PACKAGED_FINE_TUNED_REDUCTION_BYTES = 30 * 1024 * 1024
SIZE_REDUCTION_THRESHOLD_EXIT = 3
HASH_CHUNK_BYTES = 1024 * 1024


class BenchmarkIntegrityError(ValueError):
    """Raised when benchmark inputs are not the pinned, verified assets."""


class BenchmarkParityError(ValueError):
    """Raised when the checked legacy and bundle predictions differ."""


class BenchmarkConfigurationError(ValueError):
    """Raised when the process cannot use the requested controlled settings."""


@dataclass(frozen=True)
class BenchmarkAssets:
    """Loaded inputs whose source and bundle integrity was verified first."""

    legacy_models: tuple[Prf_to_Seg_Model, ...]
    ensemble: Any
    legacy_file_bytes: int
    legacy_tensor_bytes: int
    bundle_file_bytes: int
    bundle_tensor_bytes: int
    bundle_sha256: str
    source_digests: tuple[dict[str, Any], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(state: Mapping[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


def _load_legacy_models(legacy_dir: Path) -> tuple[
    tuple[Prf_to_Seg_Model, ...], tuple[dict[str, Any], ...], int, int
]:
    contract = _load_source_contract(DEFAULT_SOURCE_CONTRACT)
    states = _load_verified_members(legacy_dir, contract)
    models = []
    for state in states:
        model = Prf_to_Seg_Model().eval()
        model.load_state_dict(state, strict=True)
        models.append(model)
    source_paths = [legacy_dir / Path(entry["path"]).name for entry in contract["members"]]
    source_digests = tuple(
        {
            "member": entry["member"],
            "path": entry["path"],
            "sha256": entry["sha256"],
        }
        for entry in contract["members"]
    )
    return (
        tuple(models),
        source_digests,
        sum(path.stat().st_size for path in source_paths),
        sum(_tensor_bytes(state) for state in states),
    )


def load_verified_assets(legacy_dir: Path, bundle: Path) -> BenchmarkAssets:
    """Load only the exact source checkpoints and manifest-pinned bundle."""
    legacy_dir = Path(legacy_dir)
    bundle = Path(bundle)
    try:
        legacy_models, source_digests, legacy_file_bytes, legacy_tensor_bytes = (
            _load_legacy_models(legacy_dir)
        )
        bundle_entry = bundled_artifact("gamma-tuned-ensemble")
        expected_path = "ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors"
        if bundle_entry["path"] != expected_path:
            raise ValueError(
                "gamma-tuned-ensemble manifest path mismatch: "
                f"expected {expected_path}, got {bundle_entry['path']}"
            )
        ensemble = load_gamma_ensemble(bundle, bundle_entry["sha256"])
        bundle_tensor_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in ensemble.state_dict().values()
        )
        if bundle_tensor_bytes != BUNDLE_TENSOR_BYTES:
            raise ValueError(
                "gamma ensemble tensor bytes mismatch: "
                f"expected {BUNDLE_TENSOR_BYTES}, got {bundle_tensor_bytes}"
            )
        return BenchmarkAssets(
            legacy_models=legacy_models,
            ensemble=ensemble,
            legacy_file_bytes=legacy_file_bytes,
            legacy_tensor_bytes=legacy_tensor_bytes,
            bundle_file_bytes=bundle.stat().st_size,
            bundle_tensor_bytes=bundle_tensor_bytes,
            bundle_sha256=_sha256_file(bundle),
            source_digests=source_digests,
        )
    except BenchmarkIntegrityError:
        raise
    except Exception as error:
        raise BenchmarkIntegrityError(f"input integrity verification failed: {error}") from error


def _fixed_sigma() -> torch.Tensor:
    return torch.full(FIXED_SIGMA_SHAPE, FIXED_SIGMA_VALUE, dtype=torch.float32)


def _legacy_members(assets: BenchmarkAssets) -> torch.Tensor:
    sigma = _fixed_sigma()
    temperature = torch.full(
        (sigma.shape[0],), FIXED_TEMPERATURE_KELVIN, dtype=sigma.dtype
    )
    return torch.stack(
        [model(sigma.clone(), temperature.clone())[1] for model in assets.legacy_models]
    )


def _legacy_mean(assets: BenchmarkAssets) -> torch.Tensor:
    return _legacy_members(assets).mean(dim=0)


def _legacy_mean_std(assets: BenchmarkAssets) -> tuple[torch.Tensor, torch.Tensor]:
    members = _legacy_members(assets)
    return members.mean(dim=0), members.std(dim=0, unbiased=False)


def _new_mean(assets: BenchmarkAssets) -> torch.Tensor:
    return assets.ensemble.predict_segac(_fixed_sigma(), FIXED_TEMPERATURE_KELVIN)


def _new_mean_std(assets: BenchmarkAssets) -> tuple[torch.Tensor, torch.Tensor]:
    members = assets.ensemble.predict_segac(
        _fixed_sigma(), FIXED_TEMPERATURE_KELVIN, return_members=True
    )
    return members.mean(dim=0), members.std(dim=0, unbiased=False)


def _max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def verify_numerical_parity(assets: BenchmarkAssets) -> dict[str, float]:
    """Reject a benchmark when the bundle does not reproduce legacy outputs."""
    try:
        legacy_mean = _legacy_mean(assets)
        new_mean = _new_mean(assets)
        legacy_mean_std = _legacy_mean_std(assets)
        new_mean_std = _new_mean_std(assets)
        torch.testing.assert_close(new_mean, legacy_mean, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            new_mean_std[0], legacy_mean_std[0], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            new_mean_std[1], legacy_mean_std[1], rtol=1e-5, atol=1e-6
        )
        return {
            "mean_max_abs_error": _max_abs_error(new_mean, legacy_mean),
            "mean_std_max_abs_error": max(
                _max_abs_error(new_mean_std[0], legacy_mean_std[0]),
                _max_abs_error(new_mean_std[1], legacy_mean_std[1]),
            ),
        }
    except Exception as error:
        raise BenchmarkParityError(f"numerical parity check failed: {error}") from error


def _time_batch(workload: Callable[[], object], iterations: int) -> float:
    start_ns = time.perf_counter_ns()
    for _ in range(iterations):
        workload()
    return (time.perf_counter_ns() - start_ns) / iterations


def _workloads(assets: BenchmarkAssets) -> dict[str, Callable[[], object]]:
    return {
        "legacy_ten_member_mean": lambda: _legacy_mean(assets),
        "new_ensemble_mean": lambda: _new_mean(assets),
        "legacy_ten_member_mean_std_reference": lambda: _legacy_mean_std(assets),
        "new_ensemble_mean_std": lambda: _new_mean_std(assets),
    }


def measure_workloads(
    assets: BenchmarkAssets,
    warmup: int,
    iterations: int,
    _threads: int,
) -> dict[str, tuple[float, ...]]:
    """Return per-call batch means after warmup, collecting between cases."""
    measurements = {}
    for name, workload in _workloads(assets).items():
        gc.collect()
        for _ in range(warmup):
            workload()
        measurements[name] = tuple(
            _time_batch(workload, iterations) for _ in range(REPEATED_BATCHES)
        )
        gc.collect()
    return measurements


def _timing_report(samples: tuple[float, ...]) -> dict[str, Any]:
    if len(samples) != REPEATED_BATCHES:
        raise BenchmarkConfigurationError(
            f"expected exactly {REPEATED_BATCHES} timing batches, got {len(samples)}"
        )
    median_ns = float(statistics.median(samples))
    return {
        "batch_mean_ns": list(samples),
        "max_ns": float(max(samples)),
        "median_absolute_deviation_ns": float(
            statistics.median(abs(sample - median_ns) for sample in samples)
        ),
        "median_ns": median_ns,
        "min_ns": float(min(samples)),
    }


def _reduction_report(legacy: int, new: int) -> dict[str, int | float]:
    if legacy <= 0 or new <= 0 or new > legacy:
        raise BenchmarkIntegrityError(
            f"invalid asset sizes for reduction calculation: legacy={legacy}, new={new}"
        )
    return {
        "legacy": legacy,
        "new": new,
        "reduction_bytes": legacy - new,
        "reduction_percent": (legacy - new) * 100.0 / legacy,
    }


def environment_report(threads: int) -> dict[str, Any]:
    """Report host facts required to reproduce controlled local measurements."""
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_threads": torch.get_num_threads(),
        "threads_requested": threads,
    }


def _configure_threads(threads: int) -> None:
    if type(threads) is not int or threads < 1:
        raise BenchmarkConfigurationError("threads must be a positive integer")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError as error:
        if torch.get_num_interop_threads() != threads:
            raise BenchmarkConfigurationError(
                f"could not set PyTorch inter-op threads to {threads}"
            ) from error
    if torch.get_num_threads() != threads or torch.get_num_interop_threads() != threads:
        raise BenchmarkConfigurationError(
            f"PyTorch did not apply the requested thread count {threads}"
        )


def _build_report(
    assets: BenchmarkAssets,
    parity: Mapping[str, float],
    measurements: Mapping[str, tuple[float, ...]],
    *,
    threads: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    required_cases = {
        "legacy_ten_member_mean",
        "new_ensemble_mean",
        "legacy_ten_member_mean_std_reference",
        "new_ensemble_mean_std",
    }
    if set(measurements) != required_cases:
        raise BenchmarkConfigurationError(
            "timing cases mismatch: "
            f"expected {sorted(required_cases)}, got {sorted(measurements)}"
        )
    timings = {name: _timing_report(measurements[name]) for name in sorted(measurements)}
    legacy_mean_ns = timings["legacy_ten_member_mean"]["median_ns"]
    new_mean_ns = timings["new_ensemble_mean"]["median_ns"]
    legacy_mean_std_ns = timings["legacy_ten_member_mean_std_reference"]["median_ns"]
    new_mean_std_ns = timings["new_ensemble_mean_std"]["median_ns"]
    if new_mean_ns <= 0 or new_mean_std_ns <= 0:
        raise BenchmarkConfigurationError("timing medians must be positive")
    return {
        "configuration": {
            "iterations": iterations,
            "repeated_batches": REPEATED_BATCHES,
            "threads_requested": threads,
            "warmup": warmup,
        },
        "environment": environment_report(threads),
        "input": {
            "sigma_all_positive": True,
            "sigma_shape": list(FIXED_SIGMA_SHAPE),
            "sigma_value": FIXED_SIGMA_VALUE,
            "temperature_kelvin": FIXED_TEMPERATURE_KELVIN,
        },
        "integrity": {
            "bundle": {
                "file_bytes": assets.bundle_file_bytes,
                "sha256": assets.bundle_sha256,
                "tensor_bytes": assets.bundle_tensor_bytes,
            },
            "legacy_sources": list(assets.source_digests),
        },
        "numerical_parity": dict(parity),
        "reductions": {
            "file_bytes": _reduction_report(
                assets.legacy_file_bytes, assets.bundle_file_bytes
            ),
            "tensor_bytes": _reduction_report(
                assets.legacy_tensor_bytes, assets.bundle_tensor_bytes
            ),
        },
        "schema_version": 1,
        "speed_ratios": {
            "legacy_mean_over_new_mean": legacy_mean_ns / new_mean_ns,
            "legacy_mean_std_over_new_mean_std": legacy_mean_std_ns / new_mean_std_ns,
        },
        "timings": timings,
    }


def _write_report(output: Path, report: Mapping[str, Any]) -> None:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise BenchmarkConfigurationError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise BenchmarkConfigurationError(f"output parent does not exist: {output.parent}")
    output.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _text_report(report: Mapping[str, Any]) -> str:
    timings = report["timings"]
    ratios = report["speed_ratios"]
    return "\n".join(
        [
            "gamma-ensemble benchmark",
            "legacy ten-member mean: "
            f"{timings['legacy_ten_member_mean']['median_ns']:.3f} ns/call",
            "new ensemble mean: "
            f"{timings['new_ensemble_mean']['median_ns']:.3f} ns/call "
            f"({ratios['legacy_mean_over_new_mean']:.3f}x)",
            "legacy ten-member mean+std reference: "
            f"{timings['legacy_ten_member_mean_std_reference']['median_ns']:.3f} ns/call",
            "new ensemble mean+std: "
            f"{timings['new_ensemble_mean_std']['median_ns']:.3f} ns/call "
            f"({ratios['legacy_mean_std_over_new_mean_std']:.3f}x)",
            f"bundle sha256: {report['integrity']['bundle']['sha256']}",
        ]
    )


def run_benchmark(
    legacy_dir: Path,
    bundle: Path,
    *,
    threads: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Verify inputs, check parity, and time the four controlled workloads."""
    if warmup < 0:
        raise BenchmarkConfigurationError("warmup must be zero or greater")
    if iterations < 1:
        raise BenchmarkConfigurationError("iterations must be a positive integer")
    _configure_threads(threads)
    assets = load_verified_assets(legacy_dir, bundle)
    parity = verify_numerical_parity(assets)
    measurements = measure_workloads(assets, warmup, iterations, threads)
    return _build_report(
        assets,
        parity,
        measurements,
        threads=threads,
        warmup=warmup,
        iterations=iterations,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(
            args.legacy_dir,
            args.bundle,
            threads=args.threads,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        _write_report(args.output, report)
    except BenchmarkIntegrityError as error:
        print(f"benchmark integrity failure: {error}", file=sys.stderr)
        return 1
    except BenchmarkParityError as error:
        print(f"benchmark parity failure: {error}", file=sys.stderr)
        return 1
    except BenchmarkConfigurationError as error:
        print(f"benchmark configuration failure: {error}", file=sys.stderr)
        return 1
    print(_text_report(report))
    packaged_reduction = report["reductions"]["file_bytes"]["reduction_bytes"]
    if packaged_reduction < MIN_PACKAGED_FINE_TUNED_REDUCTION_BYTES:
        print(
            "benchmark size failure: packaged fine-tuned reduction is below "
            f"the required {MIN_PACKAGED_FINE_TUNED_REDUCTION_BYTES / (1024 * 1024):.0f} MiB",
            file=sys.stderr,
        )
        return SIZE_REDUCTION_THRESHOLD_EXIT
    if report["speed_ratios"]["legacy_mean_over_new_mean"] < MIN_MEAN_SPEEDUP:
        print(
            "benchmark speed failure: new ensemble mean is below "
            f"the required {MIN_MEAN_SPEEDUP:.1f}x speedup",
            file=sys.stderr,
        )
        return SPEED_THRESHOLD_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
