# Shared-Trunk Gamma Ensemble Design

## Status and scope

This design replaces the ten separately packaged and separately executed
fine-tuned gamma models with one strictly validated shared-trunk ensemble.
It changes the tuned activity-coefficient API so ensemble standard deviation
is returned by default, while still allowing callers to request the historical
mean-only result or one numbered fine-tuned member.

This work does not change the base gamma model, the sigma-profile or geometry
models, SMI-TED, ChemBERTa2, NRTL fitting outputs, or any external model
release. It does not create a Git tag, GitHub Release, or PyPI publication.

## Verified source facts

The source set is the ten bundled files
`src/tennetsac/ckpt_files/fine-tuned/1.ckpt` through `10.ckpt`, whose digests
are pinned at source commit
`7711c6bd3dff7e2d7cff00590062f9e1a0ca605a`.

Because the ten artifact records will leave the package manifest after
conversion, their provenance must remain in a repository-only immutable input
contract, `scripts/gamma_ensemble_sources.json`. It records source bundle
version `1.0.0`, source commit
`7711c6bd3dff7e2d7cff00590062f9e1a0ca605a`, and these exact path/digest pairs:

| Path | SHA-256 |
| --- | --- |
| `ckpt_files/fine-tuned/1.ckpt` | `134be46cce6839a7b08250750e4803c8198a4725d5973ca124db76dd488aedc1` |
| `ckpt_files/fine-tuned/2.ckpt` | `d763688bbe2898fc182ae6e9b39436a678ad892510c76b510c43c389e87d4c5b` |
| `ckpt_files/fine-tuned/3.ckpt` | `15232df78dbbce274818e00bdfff455a163fbb9ff8af1ef3ba4de73650519400` |
| `ckpt_files/fine-tuned/4.ckpt` | `bf01d2c6832b161ddf12d789dc886a3c3065a1810948d196f103a89c94f0daeb` |
| `ckpt_files/fine-tuned/5.ckpt` | `937a02283521bba254b917685f4db265c48617732f9c5bbfffb14eaaf1d38813` |
| `ckpt_files/fine-tuned/6.ckpt` | `7588bfe7a265752395373cf930ede2e721da2cb37fd7779a7b3b94d6c3590e79` |
| `ckpt_files/fine-tuned/7.ckpt` | `f4c03f8281b7ac56213e5e04123c208fc4f2030ca610684bd1cbd9aa59f7e16d` |
| `ckpt_files/fine-tuned/8.ckpt` | `0ff319a81281b4315638281283433590e6adf7b7ab17039919601747f736096d` |
| `ckpt_files/fine-tuned/9.ckpt` | `73c65639135717a3eaa7130c03ea6f85fa3f95ed1b5937b5b73611bd33378c70` |
| `ckpt_files/fine-tuned/10.ckpt` | `def4bb97274011564b33d55666bcc5d4d2bcae9c422cbf2c96529c7c364ebfcc` |

Direct state-dictionary comparison established the following facts:

- every checkpoint contains the same 39 keys;
- 33 keys, including parameters and BatchNorm buffers, are bitwise identical
  across all ten checkpoints;
- the only differing keys are
  `model_final.0.{weight,bias}`, `model_final.2.{weight,bias}`, and
  `model_final.4.{weight,bias}`;
- one model contains 3,718,748 tensor bytes;
- storing the 33 common tensors once and all ten six-tensor heads requires
  5,202,560 tensor bytes instead of 37,187,480 bytes;
- the ten source files occupy 37,324,100 bytes on disk;
- the current ten-model mean-only reference takes approximately 3.01 ms per
  call for a `(1, 51)` input with one PyTorch thread on the development host.

These observations are release inputs, not assumptions. The converter must
re-establish them before producing an artifact.

## Chosen architecture

The package will contain one
`src/tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors` file.
The file stores 93 tensors:

- 33 shared tensors under `trunk.<original-key>`; and
- 60 member tensors under
  `heads.<zero-based-member>.<original-model_final-suffix>`.

For example, source key `model_sigma.0.weight` becomes
`trunk.model_sigma.0.weight`, and source key `model_final.2.bias` from
checkpoint `7.ckpt` becomes `heads.6.2.bias`.

A new `GammaEnsemble` module owns one copy of the existing sigma, temperature,
combined, BatchNorm, and residual modules plus a `ModuleList` of ten existing
three-layer final networks. The established `Prf_to_Seg_Model` remains the
base-model implementation and retains its existing state-dictionary names and
forward contract. Shared forward logic may be extracted into a private helper,
but the base checkpoint layout must not change.

The runtime replaces its private tuple of ten full models with one
`GammaEnsemble`. No legacy ten-checkpoint fallback is provided in the new
package: silently choosing a different asset layout would make package
integrity and performance depend on accidental files.

## Reproducible conversion and bundle contract

An offline converter accepts an explicit source directory and explicit output
path. It performs these operations in order:

1. Resolve exactly `1.ckpt` through `10.ckpt`; reject missing, extra-selected,
   symlink-substituted, or non-regular source files.
2. Stream-hash every source and compare it with the corresponding immutable
   source-contract digest before deserialization.
3. Load weights on CPU with restricted PyTorch deserialization.
4. Require the exact 39-key set and require every corresponding tensor to
   have the same shape and dtype across all members.
5. Require all 33 non-`model_final` tensors to be bitwise equal across all ten
   members.
6. Require the differing-key set to be exactly the six approved
   `model_final` keys. A coincidentally equal head tensor is allowed, but no
   other key may differ.
7. Detach tensors, move them to CPU, and make them contiguous without changing
   values or dtypes.
8. Write the 93-tensor safetensors file to a same-directory temporary file,
   canonicalize its header with the proven SMI-TED canonical-header logic,
   strictly reload it, and verify every value, key, shape, dtype, and metadata
   field before atomic installation.

The converter never overwrites an existing target and removes only temporary
files it created. The reusable header canonicalizer moves from the SMI-TED
converter into a focused script helper and remains covered by the existing
SMI-TED determinism tests.

Required safetensors string metadata records:

- `asset_name=gamma-tuned-ensemble`
- `format_version=1`
- `member_count=10`
- `shared_tensor_count=33`
- `head_tensor_count=60`
- `source_manifest_bundle_version=1.0.0`

Two independent conversions from the same ten sources must produce identical
file bytes and SHA-256 digests.

## Manifest and distribution changes

The manifest remains schema version 2 because its JSON shape does not change.
Its `bundle_version` changes from `1.0.0` to `2.0.0`, reflecting the breaking
internal checkpoint layout. The ten `gamma-tuned-1` through
`gamma-tuned-10` artifact entries are replaced by exactly one entry:

- name: `gamma-tuned-ensemble`
- path: `ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors`
- distribution: `bundled`
- sha256: the observed lowercase digest from reproducible conversion

The ten old fine-tuned `.ckpt` files are removed only after the new bundle has
passed strict reload and reference-parity checks. `base.ckpt`, `geo.ckpt`, and
`prf.ckpt` remain byte-for-byte unchanged.

The archive verifier changes from forbidding every safetensors file to
allowing exactly this declared, hash-valid bundled safetensors path. It must
continue to reject any other `.safetensors` or `.pt` member, including the
external SMI-TED asset. Both wheel and sdist must contain exactly the three
unchanged `.ckpt` files and the one ensemble safetensors file.

## Loading and validation

`load_gamma_ensemble(path)` constructs the fixed ten-member architecture and
loads the safetensors file on CPU. Before returning an evaluation-mode module,
it requires:

- the packaged file SHA-256 to match the manifest;
- the exact metadata values above;
- exactly 93 expected tensor names;
- exact expected shapes and dtypes derived from the fixed architecture;
- no missing, unexpected, duplicate, or aliased tensor names; and
- strict state loading without ignored values.

Missing files, digest mismatches, invalid metadata, schema mismatches, or
strict-load failures must identify the logical package path and preserve the
original cause where applicable. The runtime must not attempt a network
download or fall back to the historical files.

## Prediction modes

The supported public `version` values become:

- `"tuned"`: all ten fine-tuned members;
- `"1"` through `"10"`: the corresponding one-based fine-tuned member; and
- `"base"`: the unchanged base model.

Unknown values retain an actionable `ValueError` that lists the accepted
forms.

For the tuned ensemble, sigma and temperature pass through the common trunk
exactly once per predictor call. All ten final networks then consume the same
shared tensor.

The mean-only path averages the ten Gibbs-energy outputs and takes one
autograd gradient with respect to sigma. Linearity makes this the gradient of
the ensemble mean while avoiding ten reverse traversals.

The statistics path retains the member dimension and uses PyTorch's supported
batched vector-Jacobian product to obtain all ten member gradients from the
one shared forward graph. The member gradients remain associated with their
heads so pure-component and mixture predictions can be paired before final
activity coefficients are calculated. A standard deviation of head Gibbs
energies is not a substitute for the required standard deviation of final
member `ln gamma` values.

All model execution is inference-only and in evaluation mode. Temperature
tensors use the sigma tensor's device and floating dtype. Batched sigma inputs
must be supported without coupling samples through training-mode BatchNorm.

## Public API

`return_std` defaults to `True` for both public activity-coefficient APIs.
This is an intentional `v0.2.0` behavior change.

```python
binary_lng(
    smiles,
    temperature,
    molefraction,
    version="tuned",
    return_std=True,
)
```

With `return_std=True`, `binary_lng` returns
`(mean_1, mean_2, std_1, std_2)`, each value being a list aligned with the
input mole-fraction grid. With `return_std=False`, it preserves the historical
two-list `(ln_gamma_1, ln_gamma_2)` result.

```python
multi_lng(
    smiles,
    temperature,
    composition,
    version="tuned",
    return_std=True,
)
```

With `return_std=True`, `multi_lng` returns `(mean, std)`, two lists aligned
with the input components. With `return_std=False`, it preserves the historical
single mean list.

The standard deviation is the population standard deviation across the ten
final per-member `ln gamma` predictions, equivalent to
`numpy.std(member_results, axis=0, ddof=0)`.

`return_std=True` is valid only for `version="tuned"`. For `"base"` or a
numbered member it raises an actionable `ValueError` instructing the caller to
pass `return_std=False`; a zero standard deviation must not be presented as an
uncertainty estimate.

Examples for member selection are:

```python
binary_lng(..., version="1", return_std=False)
multi_lng(..., version="10", return_std=False)
```

`fit_nrtl` explicitly calls `binary_lng(..., return_std=False)` and retains its
current signature and result structure. No standard-deviation fields are added
to NRTL fitting in this work.

## Numerical compatibility

The historical reference is the current ten independently loaded fine-tuned
models and current calculation order. Tests use fixed sigma profiles,
temperatures, binary grids, multicomponent compositions, and real full-model
fixtures where available.

Required comparisons are:

- every numbered member versus its original checkpoint;
- tuned mean versus the historical ten-model mean;
- tuned population standard deviation versus explicit per-model final
  `ln gamma` values and `numpy.std(..., ddof=0)`;
- `return_std=False` public outputs versus existing PyPI 0.1.10 golden values;
- unchanged base, profile, geometry, SMI-TED, and NRTL behavior.

Floating results must match with `rtol=1e-5` and `atol=1e-6`. Tensor keys,
source equality, bundle reload values, file digests, and artifact membership
use exact comparisons rather than numerical tolerances.

## Tests and performance evidence

Unit tests cover converter rejection, deterministic conversion, strict loading,
member selection, invalid version/standard-deviation combinations, return
shapes, population-standard-deviation semantics, and error causes. Hook-based
tests prove the common trunk executes exactly once per predictor call in both
mean-only and statistics modes.

Integration tests compare the generated real bundle with all ten historical
models before deleting the source files. Distribution tests build wheel and
sdist, verify all declared hashes, require the exact four packaged model
assets, and reject undeclared model weights.

A non-flaky structural gate requires no more than 5,300,000 tensor bytes for
the fine-tuned ensemble, an 85% or greater reduction from the current ten
in-memory models. The packaged fine-tuned asset must shrink by at least 30 MB.

Timing remains a local release measurement rather than a CI pass/fail threshold.
The same process, input, PyTorch thread count, warmup, and sample count compare:

1. historical ten-model mean-only latency;
2. new shared-trunk mean-only latency; and
3. new shared-trunk default mean-plus-standard-deviation latency.

The release report must disclose all three medians and speed ratios. If the new
mean-only path is not at least twice as fast on the controlled development-host
benchmark, the implementation is not ready for release even if functional
tests pass.

## Documentation and release boundary

README examples show the new default statistics result, explicit mean-only
compatibility, and numbered member selection. The package changelog or release
notes identify the default return-shape change as breaking. Model provenance
records the ten source artifact names/digests, converter commit, deterministic
bundle digest, exact tensor mapping, and measured size reduction.

Implementation may generate and commit the new project-owned bundle and remove
the ten redundant project-owned checkpoints after all local gates pass. It may
not push, tag, create or modify a GitHub Release, or publish PyPI without a
separate explicit authorization after branch review.
