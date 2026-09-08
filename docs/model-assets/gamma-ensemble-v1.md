# Gamma ensemble v1 provenance

`gamma-ensemble-v1.safetensors` is a lossless structural repack of this
project's ten existing fine-tuned gamma checkpoints. It is a project-owned
bundled asset, not an external model.

## Source inputs

- Source commit: `7711c6bd3dff7e2d7cff00590062f9e1a0ca605a`
- Source bundle version: `1.0.0`
- Source contract: `scripts/gamma_ensemble_sources.json`

| Member | Original path | SHA-256 |
| ---: | --- | --- |
| 1 | `ckpt_files/fine-tuned/1.ckpt` | `134be46cce6839a7b08250750e4803c8198a4725d5973ca124db76dd488aedc1` |
| 2 | `ckpt_files/fine-tuned/2.ckpt` | `d763688bbe2898fc182ae6e9b39436a678ad892510c76b510c43c389e87d4c5b` |
| 3 | `ckpt_files/fine-tuned/3.ckpt` | `15232df78dbbce274818e00bdfff455a163fbb9ff8af1ef3ba4de73650519400` |
| 4 | `ckpt_files/fine-tuned/4.ckpt` | `bf01d2c6832b161ddf12d789dc886a3c3065a1810948d196f103a89c94f0daeb` |
| 5 | `ckpt_files/fine-tuned/5.ckpt` | `937a02283521bba254b917685f4db265c48617732f9c5bbfffb14eaaf1d38813` |
| 6 | `ckpt_files/fine-tuned/6.ckpt` | `7588bfe7a265752395373cf930ede2e721da2cb37fd7779a7b3b94d6c3590e79` |
| 7 | `ckpt_files/fine-tuned/7.ckpt` | `f4c03f8281b7ac56213e5e04123c208fc4f2030ca610684bd1cbd9aa59f7e16d` |
| 8 | `ckpt_files/fine-tuned/8.ckpt` | `0ff319a81281b4315638281283433590e6adf7b7ab17039919601747f736096d` |
| 9 | `ckpt_files/fine-tuned/9.ckpt` | `73c65639135717a3eaa7130c03ea6f85fa3f95ed1b5937b5b73611bd33378c70` |
| 10 | `ckpt_files/fine-tuned/10.ckpt` | `def4bb97274011564b33d55666bcc5d4d2bcae9c422cbf2c96529c7c364ebfcc` |

## Conversion and layout

The converter was introduced in commit
`dfc19918d3de0f2c510a8b5899c4a788a7e7cf10`. It first verifies each source
digest, then checks that the shared tensors are byte-identical before writing a
canonical safetensors file.

```bash
conda run -n tsac_env python scripts/build_gamma_ensemble_asset.py \
  /private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908 \
  /private/tmp/gamma-ensemble-v1.safetensors \
  --source-contract scripts/gamma_ensemble_sources.json
```

The mapping contains 33 shared legacy tensors under `trunk.*` and six final
head tensors for each of ten members under `heads.0.*` through `heads.9.*`:
33 + 60 = 93 tensors. The tensor payload is exactly 5,202,560 bytes.

| Output path | Complete-file bytes | SHA-256 |
| --- | ---: | --- |
| `ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors` | 5,210,544 | `9ebd1b3b72c406c1987afd7ee842cdbc152c019148e6c97aea5e39cf74a79e22` |

## Parity verification

Run the real-asset parity gate with the preserved source directory:

```bash
TENNETSAC_LEGACY_GAMMA_DIR=/private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908 \
TENNETSAC_GAMMA_ENSEMBLE=src/tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors \
conda run -n tsac_env python -m pytest tests/integration/test_gamma_ensemble_asset.py -v
```

## Controlled benchmark evidence

Task 7 ran three controlled local measurements against the verified preserved
sources and the integrated bundle. The representative median run by new
mean-path time was r3: legacy mean 3,079,825.166 ns/call, new mean 631,969.666
ns/call (4.8734×), explicit legacy ten-member mean+std reference 3,082,728.582
ns/call, and new mean+std 1,238,649.166 ns/call (2.4888×). The exact digest,
all three runs, batch dispersions, environment, parity errors, and byte
reductions are recorded in the
[controlled benchmark report](../benchmarks/2026-09-08-gamma-ensemble.md).
