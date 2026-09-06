# Third-Party Notices

TeNNet-SAC includes modified third-party source code. The TeNNet-SAC project's
own source remains available under the root MIT license. The components below
retain their upstream terms.

## IBM materials / SMI-TED light

- Repository: https://github.com/IBM/materials
- Immutable source commit: `b16a458f37e6ce91997d3d3f6a12037971eb9f94`
- Upstream path: `models/smi_ted/inference/smi_ted_light/`
- Imported scope: `load.py`, the tokenizer logic originally contained in that
  module and now extracted to local `tokenizer.py`, and
  `bert_vocab_curated.txt`.
- License: Apache-2.0. A verbatim copy is in
  `licenses/IBM-materials-APACHE-2.0.txt`.
- Upstream NOTICE status: the upstream repository has no root NOTICE file at
  the recorded commit.

Local modifications include package-relative imports; extraction and hardening
of the tokenizer code; quieter diagnostics and progress reporting; pinned,
local-first checkpoint loading; checkpoint digest checks; and actionable error
handling for download failures.

The SMI-TED model checkpoint is not copied into this distribution. It remains
an external, revision-pinned download.

## Idiap fast-transformers

- Repository: https://github.com/idiap/fast-transformers
- Immutable source commit: `2ad36b97e64cb93862937bd21fcc9568d989561f`
- Upstream path and imported scope: the Python modules under
  `fast_transformers/`, copied locally to
  `src/tennetsac/smi_ted_light/fast_transformers/`, except upstream
  `attention/aft_attention.py`.
- License: MIT; copyright Idiap Research Institute, Angelos Katharopoulos, and
  Apoorv Vyas as stated in the upstream license. A verbatim copy is in
  `licenses/fast-transformers-MIT.txt`.
- Files not copied: native C++ and CUDA implementation files, upstream tests,
  benchmarks, packaging files, and `attention/aft_attention.py`.

The copied Python files retain their upstream copyright and license notices
where present. They are vendored as the pure-Python support needed by the
SMI-TED light implementation.

Local differences from the recorded upstream Python tree are limited to the
SMI-TED-compatible subset: attention exports are restricted to the full and
linear implementations; AFT support is omitted; builder and attention-layer
extensions for differing query/key model dimensions are absent; intermediate
output event hooks are absent; activation selection is restricted to ReLU or
GELU; and the Fourier feature helper uses `torch.qr` rather than
`torch.linalg.qr`. These differences reproduce the Python implementation
vendored with the SMI-TED light source rather than adding native acceleration.
