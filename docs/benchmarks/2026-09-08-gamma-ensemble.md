# Controlled Gamma-Ensemble Benchmark

This record measures the verified ten-checkpoint legacy ensemble against the
strict bundled shared-trunk ensemble on the local controlled host. It is a
reproducible local measurement, not a cross-machine performance claim.

## Method

Each run first verifies all ten preserved legacy SHA-256 digests from
`scripts/gamma_ensemble_sources.json`, then loads the bundle with the exact
manifest SHA-256. It checks numerical parity before timing one fixed positive
`(1, 51)` sigma profile (every entry `0.01`) at `298.15 K`.

The process sets PyTorch intra-op and inter-op thread counts to one. For each
of four cases it performs garbage collection before and after the case, 20
untimed warmup calls, then five batches of 500 calls measured with
`time.perf_counter_ns`; each batch result is the mean nanoseconds per call.
The tables report the median and median absolute deviation (MAD) over those
five batch means. The legacy mean+std reference explicitly runs all ten legacy
members and applies the population standard deviation (`ddof=0`); the new
mean+std path obtains the ensemble member tensor once and applies the same
reductions.

```bash
conda run -n tsac_env python scripts/benchmark_gamma_ensemble.py \
  --legacy-dir /private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908 \
  --bundle src/tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors \
  --threads 1 --warmup 20 --iterations 500 \
  --output /private/tmp/tennetsac-gamma-ensemble-benchmark-20260908-rN.json
```

## Environment and integrity

| Field | Observed value |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.1.2 |
| Platform | macOS-15.7.3-arm64-arm-64bit |
| Machine / processor | arm64 / arm |
| Requested / observed PyTorch threads | 1 / 1 intra-op, 1 inter-op |
| Bundle file / tensor bytes | 5,210,544 / 5,202,560 |
| Bundle SHA-256 | `9ebd1b3b72c406c1987afd7ee842cdbc152c019148e6c97aea5e39cf74a79e22` |
| Legacy ten-source file / tensor bytes | 37,324,100 / 37,187,480 |
| File reduction | 32,113,556 bytes (86.0397%) |
| Tensor reduction | 31,984,920 bytes (86.0099%) |
| Numerical-parity maximum absolute errors | mean `4.291534423828125e-06`; mean/std `4.76837158203125e-06` |

The file reduction is 30.6259 MiB, so it exceeds the 30 MiB packaged
fine-tuned-asset requirement.

## All controlled runs

All values are ns/call, formatted as median ± MAD. `mean ratio` is legacy
mean/new mean; `mean+std ratio` is legacy ten-member reference/new mean+std.

| Run | Legacy mean | New mean | Mean ratio | Legacy mean+std reference | New mean+std | Mean+std ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | 3,105,850.334 ± 20,481.082 | 624,786.750 ± 50.418 | 4.9711× | 3,083,254.082 ± 29,137.418 | 1,202,448.084 ± 611.168 | 2.5641× |
| r2 | 3,113,439.834 ± 7,702.832 | 637,173.750 ± 8,329.584 | 4.8863× | 3,129,582.334 ± 8,934.416 | 1,270,194.584 ± 1,812.582 | 2.4639× |
| r3 | 3,079,825.166 ± 111.168 | 631,969.666 ± 563.084 | 4.8734× | 3,082,728.582 ± 3,329.998 | 1,238,649.166 ± 22,172.916 | 2.4888× |

`r3` is the median run when ordered by the new mean-path median. Its
4.8734× mean speedup clears the required 2.0× gate. Mean+std is reported
without an additional threshold, as intended.

The temporary JSON evidence files retain every batch mean, min/max, full
environment block, verified source digests, and the exact command parameters:
`/private/tmp/tennetsac-gamma-ensemble-benchmark-20260908-r1.json`,
`/private/tmp/tennetsac-gamma-ensemble-benchmark-20260908-r2.json`, and
`/private/tmp/tennetsac-gamma-ensemble-benchmark-20260908-r3.json`.
