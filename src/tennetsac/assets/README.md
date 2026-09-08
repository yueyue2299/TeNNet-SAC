# Bundled and reserved model assets

`ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors` is a bundled,
project-owned TeNNet-SAC gamma-ensemble asset. It is not an external model and
does not need a third-party model notice.

No external model weights are included in this package. In particular, the
SMI-TED inference-only safetensors asset is downloaded from its immutable model
release into the user cache, not placed in this directory or a package archive.

`assets/chemberta2/` is reserved for a possible future offline ChemBERTa2
bundle and is intentionally empty in this release. Review ChemBERTa2's
redistribution terms in a separate design/release process before adding files
there.
