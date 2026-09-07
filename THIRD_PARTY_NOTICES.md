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

### Derived SMI-TED inference asset

The default SMI-TED asset is a TeNNet-SAC-distributed inference-only derivative
named `smi-ted-light-inference-v1.safetensors`; it is not copied into this
package distribution. Its immutable parent is the historical repository
`ibm/materials.smi-ted`, now canonicalized as
`ibm-research/materials.smi-ted`, at revision
`414c3ea0a8603ef49d1c5bb3db336e09877c01ce`, from checkpoint
`smi-ted-Light_40.pt` with SHA-256
`baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375`.

The derivative preserves the original float32 values only for the embedding
path: token embedding, encoder blocks, and the autoencoder encoder/projector.
The reconstruction decoder, language-model heads, and unused `net` are pruned.
The resulting state contains exactly 224 tensors. It is distributed by
TeNNet-SAC under the retained upstream Apache-2.0 terms, together with the
verbatim Apache license in `licenses/IBM-materials-APACHE-2.0.txt` and release
provenance material. This derivative is not an IBM-published checkpoint and
does not imply IBM endorsement.

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
