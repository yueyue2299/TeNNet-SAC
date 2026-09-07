# SMI-TED Inference-Only Asset Design

## Context

TeNNet-SAC uses SMI-TED only to encode canonicalized SMILES into 768-dimensional
embeddings. The current pinned SMI-TED checkpoint is the upstream
`smi-ted-Light_40.pt` file requested through the historical Hugging Face ID
`ibm/materials.smi-ted` (now canonicalized as
`ibm-research/materials.smi-ted`) at revision
`414c3ea0a8603ef49d1c5bb3db336e09877c01ce`, with SHA-256
`baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375`.
It is 1,155,124,350 bytes on disk and uses PyTorch's pickle-based checkpoint
format.

The complete checkpoint supports encoding and molecular-string reconstruction,
but TeNNet-SAC never calls the reconstruction path. A read-only probe of the
checkpoint and current runtime established the following split:

| State component | Tensor bytes | Used by TeNNet-SAC |
|---|---:|---|
| `encoder.tok_emb.*` and `encoder.blocks.*` | 177,695,232 | Yes |
| `decoder.autoencoder.encoder.*` | 478,946,304 | Yes |
| `encoder.lang_model.*` | 9,719,808 | No |
| `decoder.autoencoder.decoder.*` | 478,946,304 | No |
| `decoder.lang_model.*` | 9,719,808 | No |

The required state is exactly 656,641,536 tensor bytes (626.22 MiB). The
current loader also instantiates an uncheckpointed `net` with 4,727,812 bytes of
parameters even though the embedding path never calls it. Removing the language
heads, reconstruction decoder, and `net` produced bitwise-identical embeddings
for ethanol, benzene, and acetic acid (`max_abs_diff == 0.0`).

This design replaces the default full checkpoint with a reproducibly derived,
inference-only safetensors asset distributed through an independent GitHub
Release. It reduces both network/storage cost and instantiated model memory.

## Goals

1. Reduce the default SMI-TED model download from approximately 1.15 GB to an
   approximately 657 MB inference-only asset without changing embeddings or
   TeNNet-SAC predictions.
2. Instantiate only modules used by the embedding path, so RAM and accelerator
   memory fall with the artifact size.
3. Replace pickle loading on the default path with strict safetensors loading.
4. Make conversion from the pinned IBM checkpoint reproducible and auditable.
5. Store the derived model as an immutable, independently versioned GitHub
   Release asset rather than inside the PyPI wheel.
6. Support first-use download, explicit prefetch, offline operation, cache
   integrity, and an explicit local checkpoint override.
7. Preserve a temporary compatibility path for the exact legacy IBM `.pt`
   checkpoint.

## Non-goals

- Do not quantize or cast the tensors to float16/bfloat16. Version 1 preserves
  the original float32 values exactly.
- Do not bundle the derived model in the wheel or source distribution.
- Do not change SMI-TED tokenization, canonicalization, maximum sequence length,
  or embedding dimensionality.
- Do not implement the ten-model shared-trunk ensemble optimization here.
- Do not bundle ChemBERTa2 here. This work establishes a model-asset pattern that
  the later ChemBERTa2 task can reuse.
- Do not preserve SMI-TED decode or molecular-string reconstruction behavior on
  the new inference-only class. The explicit legacy loader remains available
  during the transition for that upstream behavior.
- Do not create, upload, publish, replace, or delete a GitHub Release without a
  separate explicit authorization from the repository owner.

## Selected Approach

Create a dedicated inference-only model class and a single safetensors state
file. This is preferred over loading a partial state into the existing full
class because the latter still allocates the unused decoder and heads. Splitting
the encoder and projector into multiple assets is also rejected: both are needed
for every TeNNet-SAC embedding, so multiple downloads and hashes add complexity
without enabling useful lazy loading.

## Runtime Architecture

The new runtime path is:

```text
first TeNNet-SAC prediction
    -> resolve SMI-TED asset
        -> explicit TENNETSAC_SMI_TED_CHECKPOINT, if set
            -> .safetensors: strict inference-only loader
            -> exact legacy .pt: legacy loader + FutureWarning
        -> otherwise, versioned TeNNet-SAC cache
            -> valid cache hit: use it
            -> cache miss: download pinned GitHub Release asset
            -> hash mismatch: reject; never expose partial file as valid
    -> construct SmiTedInferenceModel
        -> token embedding
        -> 12-layer rotary transformer encoder
        -> autoencoder encoder/projector
    -> return 768-dimensional embedding
    -> cached TeNNet-SAC Runtime reuses the model
```

Plain `import tennetsac` remains offline and must not import torch,
transformers, safetensors, the asset downloader, or the model runtime.

### Inference-only class

The implementation introduces a small inference-focused module rather than
teaching the reconstruction class to tolerate arbitrary missing keys. Its only
learned components are:

- the existing token embedding;
- the existing rotary transformer encoder blocks and their buffers; and
- the encoder half of the continuous-token autoencoder, exposed as
  `projector`.

The saved key namespace is normalized to:

- `encoder.tok_emb.*`;
- `encoder.blocks.*`; and
- `projector.*` (mapped from upstream
  `decoder.autoencoder.encoder.*`).

The inference state contains exactly 224 tensors and 656,641,536 tensor bytes.
The class has 656,541,696 parameter bytes; the remaining 99,840 state bytes are
registered buffers required by the encoder. It must have no `lang_model`,
autoencoder decoder, reconstruction head, or `net` attributes.

Loading is strict: the loader computes the exact expected key set from a newly
constructed inference model, compares it with the file, and rejects missing or
unexpected keys before calling `load_state_dict(..., strict=True)`. Tensor
shapes, dtypes, state byte count, architecture metadata, and vocab size are also
validated. A partially compatible checkpoint must never load silently.

## Derived Asset Format

The independently versioned model release is:

- repository: `yueyue2299/TeNNet-SAC`;
- release tag: `model-smi-ted-light-v1`;
- primary asset: `smi-ted-light-inference-v1.safetensors`;
- download URL:
  `https://github.com/yueyue2299/TeNNet-SAC/releases/download/model-smi-ted-light-v1/smi-ted-light-inference-v1.safetensors`.

The release also contains:

- `SHA256SUMS` covering every release asset;
- `IBM-materials-APACHE-2.0.txt`; and
- `SMI_TED_INFERENCE_PROVENANCE.md` describing the parent checkpoint, conversion
  command, included prefixes, excluded prefixes, architecture, and local
  verification results.

The safetensors metadata records string representations of:

- format name and format version `1`;
- upstream repository, revision, filename, and parent SHA-256;
- pruning-rule version `1`;
- included upstream prefixes;
- normalized key mapping;
- `n_layer=12`, `n_head=12`, `n_embd=768`, and `max_len=202`;
- vocab identity and expected size `2393`; and
- state tensor count and tensor byte count.

The safetensors file cannot contain its own SHA-256 because that would be a
self-referential digest. Its exact digest is instead recorded in the package
manifest, release provenance document, and `SHA256SUMS`.

The Git repository contains the converter, verifier, metadata schema, licenses,
and provenance template, but neither the 1.15 GB parent checkpoint nor the
approximately 657 MB derived file.

## Reproducible Conversion

`scripts/build_smi_ted_inference_asset.py` accepts:

- an explicit path to the parent `.pt` checkpoint;
- an explicit output directory; and
- no network credentials or implicit download behavior.

Before deserialization it streams the parent file through SHA-256 and requires
the exact pinned IBM digest. It loads the verified file on CPU with PyTorch's
restricted weights-only loader. If the installed PyTorch cannot read the known
checkpoint through the restricted loader, conversion stops with an actionable
version/error message rather than falling back to unrestricted pickle loading.

The converter:

1. validates the parent top-level structure and required architecture values;
2. selects only `encoder.tok_emb.*`, `encoder.blocks.*`, and
   `decoder.autoencoder.encoder.*`;
3. rejects any missing required prefix, duplicate normalized key, unsupported
   tensor type, or architecture mismatch;
4. maps the projector prefix to `projector.*`;
5. writes float32 tensors with safetensors metadata;
6. reloads the completed file through the production inference loader;
7. verifies exact key count, tensor byte count, shapes, dtypes, and metadata;
8. writes the release provenance document and `SHA256SUMS`; and
9. prints the exact artifact path, file size, and SHA-256.

Output is written to a new output directory or fails if a target already exists.
The converter never overwrites an existing candidate or release artifact.

## Model Manifest Version 2

Changing the external asset contract requires `model_manifest.json`
`schema_version` 2. The thirteen bundled TeNNet-SAC checkpoints and
`bundle_version` remain unchanged because their bytes do not change.

The SMI-TED entry records:

- `name` and `distribution: "external"`;
- `format: "safetensors"` and `format_version: 1`;
- concrete GitHub repository, release tag, HTTPS URL, and asset filename;
- the exact lowercase SHA-256 generated by the converter;
- state tensor count `224` and state bytes `656641536`;
- required architecture values;
- both the historical requested and current canonical parent Hugging Face
  repository IDs, plus revision, filename, and SHA-256;
- pruning-rule version and included source prefixes; and
- the legacy environment override name.

The schema validator uses exact allowed/required keys and validates the URL,
release tag, filename, digests, integers, prefix list, and nested parent record.
The exact derived digest is inserted only after the local artifact has been
created and independently verified; a manifest cannot be committed until the
real digest is available.

ChemBERTa2 retains its current external record in schema version 2. A later
offline ChemBERTa2 design can extend the same GitHub asset structure through a
new schema version if its artifact requirements differ.

## Cache and Download Manager

The cache root is resolved in this order:

1. `TENNETSAC_CACHE_DIR`, when explicitly set;
2. `platformdirs.user_cache_path("tennetsac")` otherwise.

The default asset path is
`CACHE_ROOT/models/smi-ted-light/v1/smi-ted-light-inference-v1.safetensors`,
where `CACHE_ROOT` is the resolved value above.
The dependencies `platformdirs` and `filelock` are declared directly rather
than relying on transitive installations.

### Cache resolution

- An explicit `TENNETSAC_SMI_TED_CHECKPOINT` always wins and is never modified
  or deleted by TeNNet-SAC.
- A cache hit is SHA-256 verified once per process before safetensors loading.
  The cached TeNNet-SAC runtime avoids repeating both validation and loading.
- A corrupt cache-owned file is removed while holding the per-asset lock. In
  online mode it is downloaded again; in offline mode the loader reports the
  removed corrupt path and stops.
- The downloader takes a file lock adjacent to the target and rechecks the cache
  after acquiring it, preventing duplicate 657 MB downloads between processes.
- Downloads stream into a unique same-directory `.part` file while computing
  SHA-256. Successful verification is followed by flush/fsync and atomic
  `os.replace` to the final path.
- Exceptions and signals trigger best-effort removal of only that process's
  partial file. A partial file is never considered a cache hit.
- The configured and final redirected URLs must both use HTTPS. Content length
  may be used for progress but is not trusted as an integrity check.

`TENNETSAC_OFFLINE=1` disables GitHub network access. Missing-cache errors name
the expected path, immutable release identity, explicit checkpoint override,
and the prefetch command.

### User commands

The package provides:

```text
python -m tennetsac.model_assets download smi-ted-light
python -m tennetsac.model_assets verify smi-ted-light
```

`download` resolves/downloads/verifies the configured asset and prints its final
path and digest. `verify` performs no network access, validates the configured
override or cache file, and returns a nonzero exit code with an actionable error
if the asset is absent or invalid. These commands do not load the neural model.

## Legacy Compatibility

If `TENNETSAC_SMI_TED_CHECKPOINT` explicitly names a `.pt` file, the loader:

1. requires the exact pinned parent SHA-256 before deserialization;
2. emits a visible `FutureWarning` explaining that the legacy full checkpoint
   uses more memory and will be removed in the next major package version;
3. invokes the existing full SMI-TED loader; and
4. preserves current embedding behavior.

The package never automatically searches the Hugging Face cache for the legacy
file and never automatically downloads it. Unknown checkpoint formats and a
legacy file with any other digest are rejected. Explicit override failures do
not delete or rename user-owned files.

## Failure Handling

Every asset error identifies the model, release tag, URL or local path, expected
digest, and recovery action. In particular:

- offline cache miss: show the prefetch command and override variable;
- HTTP/redirect failure: retain the cause and show the immutable release URL;
- lock timeout: name the lock and cache target;
- disk/write/fsync failure: remove the process's partial file and retain cause;
- hash mismatch: report expected/actual digests, reject the file, and never
  rename it to the final cache path;
- safetensors metadata/key/shape/dtype mismatch: reject before model use;
- explicit legacy digest mismatch: reject before `torch.load`;
- release asset missing: package release is blocked until the model release is
  corrected through a new model version.

No failure is allowed to fall back from the new asset to an unpinned network
source.

## Licensing and Provenance

The upstream SMI-TED model card declares Apache-2.0. The derived asset retains
the upstream license and attribution. Repository and model-release materials
include the exact Apache-2.0 text, immutable parent repository/revision,
checkpoint filename and digest, explicit modification/pruning notice, and the
converter commit.

The derived artifact is identified as a TeNNet-SAC inference-only derivative,
not as an IBM-published checkpoint. Its name must not imply IBM endorsement.
`THIRD_PARTY_NOTICES.md` is updated to describe both the adapted source and the
derived model asset.

## Testing Strategy

### Unit and package tests

- Synthetic state fixtures prove the converter includes only the allowed
  prefixes, maps projector keys correctly, accepts the exact known excluded
  reconstruction prefixes, and rejects missing included keys, unknown source
  prefixes, duplicate normalized keys, or non-tensor content.
- Schema tests compare the complete version 2 manifest and reject malformed URL,
  hash, format, architecture, parent, and pruning records.
- Tiny local file URLs or a local HTTP fixture exercise real cache writes,
  locking, rechecks, atomic replacement, offline mode, interrupted downloads,
  redirect rules, and hash mismatches without external network access.
- Loader tests require strict keys, shapes, dtypes, metadata, 224 tensors, and
  656,641,536 state bytes.
- Structural tests prove the inference model does not instantiate language
  heads, the reconstruction decoder, or `net`.
- CLI tests assert exit codes and observable cache behavior rather than source
  strings.
- Legacy tests use the exact verified checkpoint fixture when available, require
  the visible warning, and reject wrong hashes before pickle loading.
- Plain package import remains offline and lazy.

### Integration tests

Integration tests use the locally generated or downloaded real v1 asset and are
excluded from ordinary offline CI unless the cache is provisioned. They prove:

- CPU SMI-TED embeddings are bitwise identical to the pinned full checkpoint
  for a fixed representative SMILES corpus;
- embedding shape and invalid-SMILES placement are unchanged;
- TeNNet-SAC profile, binary, and multicomponent predictions match the existing
  PyPI 0.1.10 golden fixture at the established tolerance;
- the real artifact has the manifest digest, metadata, 224 tensors, and exact
  tensor byte count;
- default loading never calls `torch.load` or constructs unused modules; and
- a clean environment can download the public GitHub Release asset, verify it,
  and complete one prediction.

Release verification fails unless the public model asset is reachable and the
clean-download integration smoke passes.

## Rollout

1. Implement converter, schema, inference loader, cache manager, CLI, and tests
   on `codex/smi-ted-inference-asset`.
2. Convert the locally cached parent checkpoint into the release candidate and
   record its exact digest in manifest version 2.
3. Complete task-scoped and whole-branch review without publishing externally.
4. Merge the reviewed code and push it only with repository-owner approval.
5. Create `model-smi-ted-light-v1` from the reviewed converter commit and upload
   the safetensors file, checksum manifest, Apache license, and provenance file,
   only after separate explicit authorization.
6. Enable immutable-release protection where available and publish the model
   release.
7. Run the clean public-download integration smoke against the release URL.
8. Only after that smoke passes, create the package `v0.2.0` tag and draft
   package release; PyPI publication remains its existing protected manual step.

The model release is independent of ordinary package releases. Package updates
continue referencing v1 until model content changes.

## Rollback and Model Updates

Published model assets and tags are never replaced or moved. If v1 is defective,
create `model-smi-ted-light-v2`, update the package manifest to the new URL and
digest, repeat all parity/download tests, and release a package patch version.
Existing package versions continue resolving the exact asset they declared.

Any future change to included keys, tensor values/dtypes, architecture, vocab,
or parent checkpoint increments the model release version and the pruning-format
version when the on-disk contract changes. Quantization, if ever desired, is a
separate model variant with separate accuracy criteria rather than an in-place
replacement for float32 v1.

## Acceptance Criteria

- The derived artifact contains exactly 224 tensors and 656,641,536 tensor
  bytes, with no decoder, language-head, or `net` state.
- The default runtime instantiates only 656,541,696 parameter bytes for SMI-TED.
- Representative CPU embeddings are bitwise identical to the pinned parent;
  end-to-end TeNNet-SAC outputs pass existing golden tolerances.
- The default path uses safetensors, strict validation, a pinned GitHub Release
  URL, and exact SHA-256 verification.
- Cache download is locked and atomic; offline and interrupted operations leave
  no apparently valid partial file.
- Explicit exact legacy `.pt` override works with a visible warning; automatic
  Hugging Face checkpoint download is removed.
- Wheel and sdist remain free of large SMI-TED weights.
- The model asset is published and clean-download tested before any package
  release that references it.
- No external release or publish action occurs without explicit authorization.
