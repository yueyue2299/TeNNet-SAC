"""Contract tests for the controlled gamma-ensemble benchmark CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import benchmark_gamma_ensemble as benchmark


@pytest.fixture(autouse=True)
def isolate_process_global_thread_control(monkeypatch):
    """The CLI's fresh-process thread setup is outside these mocked CLI tests."""
    monkeypatch.setattr(benchmark, "_configure_threads", lambda _threads: None)


def _assets() -> benchmark.BenchmarkAssets:
    """Return independently specified, already-verified benchmark evidence."""
    return benchmark.BenchmarkAssets(
        legacy_models=(),
        ensemble=object(),
        legacy_file_bytes=52_000_000,
        legacy_tensor_bytes=52_025_600,
        bundle_file_bytes=5_210_544,
        bundle_tensor_bytes=5_202_560,
        bundle_sha256="9ebd1b3b72c406c1987afd7ee842cdbc152c019148e6c97aea5e39cf74a79e22",
        source_digests=(
            {
                "member": 1,
                "path": "ckpt_files/fine-tuned/1.ckpt",
                "sha256": "1" * 64,
            },
        ),
    )


def _measurements(*, new_mean_median_ns: float = 200.0) -> dict[str, tuple[float, ...]]:
    """Stable batches expose report calculation without measuring real models."""
    return {
        "legacy_ten_member_mean": (500.0, 505.0, 495.0, 500.0, 500.0),
        "new_ensemble_mean": (
            new_mean_median_ns,
            new_mean_median_ns + 5.0,
            new_mean_median_ns - 5.0,
            new_mean_median_ns,
            new_mean_median_ns,
        ),
        "legacy_ten_member_mean_std_reference": (750.0, 755.0, 745.0, 750.0, 750.0),
        "new_ensemble_mean_std": (250.0, 255.0, 245.0, 250.0, 250.0),
    }


def _argv(output: Path) -> list[str]:
    return [
        "--legacy-dir",
        "/verified/legacy",
        "--bundle",
        "/verified/gamma-ensemble-v1.safetensors",
        "--threads",
        "1",
        "--warmup",
        "20",
        "--iterations",
        "500",
        "--output",
        str(output),
    ]


def _patch_successful_benchmark(monkeypatch, *, new_mean_median_ns: float = 200.0):
    monkeypatch.setattr(benchmark, "load_verified_assets", lambda *_args: _assets())
    monkeypatch.setattr(
        benchmark,
        "verify_numerical_parity",
        lambda _assets: {
            "mean_max_abs_error": 0.0,
            "mean_std_max_abs_error": 0.0,
        },
    )
    monkeypatch.setattr(
        benchmark,
        "measure_workloads",
        lambda *_args: _measurements(new_mean_median_ns=new_mean_median_ns),
    )
    monkeypatch.setattr(benchmark, "environment_report", lambda _threads: {"python": "test"})


def test_cli_writes_a_deterministic_complete_evidence_schema(monkeypatch, tmp_path):
    """Breaks if the CLI drops fixed inputs, evidence, or timing relationships."""
    _patch_successful_benchmark(monkeypatch)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert benchmark.main(_argv(first)) == 0
    assert benchmark.main(_argv(second)) == 0

    assert first.read_bytes() == second.read_bytes()
    report = json.loads(first.read_text(encoding="utf-8"))
    assert set(report) == {
        "configuration",
        "environment",
        "input",
        "integrity",
        "numerical_parity",
        "reductions",
        "schema_version",
        "speed_ratios",
        "timings",
    }
    assert report["schema_version"] == 1
    assert report["input"] == {
        "sigma_all_positive": True,
        "sigma_shape": [1, 51],
        "sigma_value": 0.01,
        "temperature_kelvin": 298.15,
    }
    assert report["configuration"] == {
        "iterations": 500,
        "repeated_batches": 5,
        "threads_requested": 1,
        "warmup": 20,
    }
    assert report["integrity"]["bundle"] == {
        "file_bytes": 5_210_544,
        "sha256": "9ebd1b3b72c406c1987afd7ee842cdbc152c019148e6c97aea5e39cf74a79e22",
        "tensor_bytes": 5_202_560,
    }
    assert report["timings"]["legacy_ten_member_mean"]["median_ns"] == 500.0
    assert report["timings"]["new_ensemble_mean"]["median_ns"] == 200.0
    assert report["timings"]["new_ensemble_mean_std"]["median_ns"] == 250.0
    assert report["timings"]["legacy_ten_member_mean_std_reference"]["median_ns"] == 750.0
    assert report["speed_ratios"] == {
        "legacy_mean_over_new_mean": 2.5,
        "legacy_mean_std_over_new_mean_std": 3.0,
    }
    assert report["reductions"]["file_bytes"] == {
        "legacy": 52_000_000,
        "new": 5_210_544,
        "reduction_bytes": 46_789_456,
        "reduction_percent": pytest.approx(89.97972307692308),
    }


def test_cli_rejects_failed_input_integrity_before_timing(monkeypatch, tmp_path, capsys):
    """Breaks if a source or bundle integrity error can be benchmarked anyway."""
    calls = []

    def reject_assets(*_args):
        raise benchmark.BenchmarkIntegrityError("legacy source sha256 mismatch")

    monkeypatch.setattr(benchmark, "load_verified_assets", reject_assets)
    monkeypatch.setattr(
        benchmark,
        "measure_workloads",
        lambda *_args: calls.append("timed") or _measurements(),
    )
    output = tmp_path / "rejected.json"

    assert benchmark.main(_argv(output)) == 1
    assert calls == []
    assert not output.exists()
    assert "integrity" in capsys.readouterr().err.lower()


def test_cli_rejects_an_alternate_source_contract_option(monkeypatch, tmp_path, capsys):
    """The benchmark contract must be the immutable repository-owned file."""
    _patch_successful_benchmark(monkeypatch)
    alternate_contract = tmp_path / "alternate-sources.json"
    alternate_contract.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        benchmark.main(
            [
                *_argv(tmp_path / "blocked.json"),
                "--source-contract",
                str(alternate_contract),
            ]
        )

    assert exit_info.value.code == 2
    assert "unrecognized arguments: --source-contract" in capsys.readouterr().err


def test_cli_returns_a_distinct_failure_when_mean_speedup_misses_threshold(
    monkeypatch, tmp_path
):
    """Breaks if a too-slow new mean path is reported as a successful benchmark."""
    _patch_successful_benchmark(monkeypatch, new_mean_median_ns=251.0)
    output = tmp_path / "too-slow.json"

    assert benchmark.main(_argv(output)) == benchmark.SPEED_THRESHOLD_EXIT
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["speed_ratios"]["legacy_mean_over_new_mean"] == pytest.approx(
        500.0 / 251.0
    )
    assert report["speed_ratios"]["legacy_mean_std_over_new_mean_std"] == 3.0
