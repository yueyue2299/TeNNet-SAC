# SMI-TED Inference-Only Asset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TeNNet-SAC's default 1.15 GB SMI-TED checkpoint with a reproducibly derived, strictly validated, approximately 657 MB inference-only safetensors asset delivered from an immutable GitHub Release.

**Architecture:** Add a dedicated SMI-TED inference class containing only the token embedding, rotary transformer blocks, and continuous-token projector. A lightweight asset manager resolves an explicit override or a versioned local cache, verifies SHA-256, and performs locked atomic HTTPS downloads; the old full `.pt` model remains available only through an exact-hash explicit override during the transition.

**Tech Stack:** Python 3.10-3.12, PyTorch 2.1.2+, safetensors, platformdirs, filelock, urllib, pytest, GitHub Releases.

**Spec:** `docs/superpowers/specs/2026-09-07-smi-ted-inference-asset-design.md`

## Global Constraints

- Work only on `codex/smi-ted-inference-asset`; do not mix in the ten-model shared-trunk optimization or ChemBERTa2 bundling.
- Preserve the exact upstream parent identity: historical repository `ibm/materials.smi-ted`, canonical repository `ibm-research/materials.smi-ted`, revision `414c3ea0a8603ef49d1c5bb3db336e09877c01ce`, filename `smi-ted-Light_40.pt`, SHA-256 `baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375`.
- Preserve float32 tensor values; do not quantize, cast, or alter tokenizer/canonicalization behavior.
- The derived state must contain exactly 224 tensors and 656,641,536 tensor bytes; the inference model must contain 656,541,696 parameter bytes and 99,840 buffer bytes.
- Production architecture is exactly `n_layer=12`, `n_head=12`, `n_embd=768`, `max_len=202`, `num_feats=32`, and vocabulary size `2393`.
- The packaged vocabulary SHA-256 is `8576b60e838336837f9e894457ef144f440c4d31bc8ceda2315fb5b28a6dfd95`.
- The default model identity is repository `yueyue2299/TeNNet-SAC`, tag `model-smi-ted-light-v1`, asset `smi-ted-light-inference-v1.safetensors`.
- Default loading must never call `torch.load`, follow a mutable branch, or fall back to Hugging Face.
- An explicit legacy `.pt` override must be checked against the exact parent digest before restricted deserialization and must emit `FutureWarning`.
- `TENNETSAC_OFFLINE=1` forbids GitHub network access. `TENNETSAC_CACHE_DIR` overrides the platform cache root. `TENNETSAC_SMI_TED_CHECKPOINT` overrides both.
- Plain `import tennetsac` must not import torch, transformers, safetensors, platformdirs, filelock, or any download code.
- Do not place `.pt` or `.safetensors` weights in Git, wheel, or sdist.
- Do not create, upload, replace, publish, or delete a GitHub Release without a new, explicit repository-owner authorization at the release gate in Task 8.
- Use `conda run -n tsac_env` for local implementation and verification commands in this workspace.

---

## File Structure

### Model contract and runtime

- `src/tennetsac/smi_ted_light/asset_contract.py`: immutable parent, architecture, prefix-map, metadata, and byte-count constants shared by conversion and loading.
- `src/tennetsac/smi_ted_light/inference.py`: inference-only modules, embedding path, safetensors metadata/state validation, and strict loader.
- `src/tennetsac/smi_ted_light/load.py`: retained upstream full model plus explicit, hash-pinned legacy `.pt` loader; no network behavior.
- `src/tennetsac/_model_assets.py`: torch-free cache path, hashing, locking, HTTPS streaming download, offline handling, and override resolution.
- `src/tennetsac/model_assets.py`: torch-free `download` and `verify` command-line interface.
- `src/tennetsac/utils/embedding.py`: checkpoint-format dispatch and the existing callable embedder adapter.
- `src/tennetsac/runtime.py`: manifest-driven SMI-TED resolution during cached runtime construction.
- `src/tennetsac/_manifest_schema.py`: exact schema-version-2 external-asset validation.
- `src/tennetsac/model_manifest.json`: immutable SMI-TED Release URL, generated digest, architecture, parent, and pruning record.

### Conversion and release material

- `scripts/build_smi_ted_inference_asset.py`: offline, no-overwrite converter and verifier for the exact parent checkpoint.
- `tests/integration/test_smi_ted_asset.py`: real parent/derived embedding parity, structure, and digest checks.
- `THIRD_PARTY_NOTICES.md`: derived-weight attribution and modification notice.
- `licenses/IBM-materials-APACHE-2.0.txt`: unchanged license copied into the model release output.

### Tests and packaging

- `tests/test_smi_ted_inference.py`: inference structure, embedding semantics, strict keys/shapes/dtypes/metadata, and no-pickle default.
- `tests/test_smi_ted_converter.py`: prefix selection, normalization, parent verification, output safety, checksum, and provenance.
- `tests/test_model_assets.py`: cache, override, offline, locking, redirects, atomic writes, corruption, and CLI behavior.
- `tests/test_smi_ted_loading.py`: exact legacy override and format dispatch.
- `tests/test_model_manifest.py`: complete literal v2 manifest and malformed-record rejection.
- `tests/test_embeddings.py`: embedder-to-loader interface.
- `tests/test_runtime.py`: runtime resolution and actionable initialization failures.
- `tests/test_package_metadata.py`: plain-import isolation.
- `tests/test_distribution.py` and `scripts/verify_distribution.py`: weights remain outside release archives.
- `tests/test_licensing.py`, `tests/test_documentation.py`, `README.md`, and `src/tennetsac/assets/README.md`: user workflow, attribution, and offline/prefetch documentation.
- `.github/workflows/release-build.yml`: block a package release unless the immutable public model asset passes a clean-download smoke test.

---

### Task 1: Define the asset contract and inference-only architecture

**Files:**
- Create: `src/tennetsac/smi_ted_light/asset_contract.py`
- Create: `src/tennetsac/smi_ted_light/inference.py`
- Test: `tests/test_smi_ted_inference.py`

**Interfaces:**
- Consumes: `MolTranBertTokenizer`, `RotateEncoderBuilder`, `GeneralizedRandomFeatures`, `LengthMask`, and `normalize_smiles` from the existing SMI-TED implementation.
- Produces: `SMI_TED_LIGHT_CONTRACT`, `SmiTedInferenceEncoder`,
  `production_config() -> dict[str, int | float]`, `SmiTedProjector`, and
  `SmiTedInferenceModel(tokenizer, config=None)` with `tokenize()`,
  `extract_embeddings()`, and `encode()`.

- [ ] **Step 1: Write failing contract and structure tests**

Create `tests/test_smi_ted_inference.py` with a tiny tokenizer/config fixture and assertions that the contract contains the exact production constants, while the model exposes only the required modules:

```python
def test_production_contract_is_exact():
    from tennetsac.smi_ted_light.asset_contract import SMI_TED_LIGHT_CONTRACT

    assert SMI_TED_LIGHT_CONTRACT.architecture == {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "max_len": 202,
        "num_feats": 32,
    }
    assert SMI_TED_LIGHT_CONTRACT.vocab_size == 2393
    assert SMI_TED_LIGHT_CONTRACT.tensor_count == 224
    assert SMI_TED_LIGHT_CONTRACT.tensor_bytes == 656_641_536


def test_inference_model_has_no_reconstruction_modules(tiny_tokenizer, tiny_config):
    from tennetsac.smi_ted_light.inference import SmiTedInferenceModel

    model = SmiTedInferenceModel(tiny_tokenizer, tiny_config)
    assert set(dict(model.named_children())) == {"encoder", "projector"}
    assert not hasattr(model.encoder, "lang_model")
    assert not hasattr(model, "decoder")
    assert not hasattr(model, "net")
```

- [ ] **Step 2: Run the focused tests and confirm the modules are absent**

Run: `conda run -n tsac_env python -m pytest tests/test_smi_ted_inference.py -v`

Expected: collection fails with `ModuleNotFoundError` for `asset_contract` or `inference`.

- [ ] **Step 3: Add immutable contract values**

Implement a frozen dataclass and one constant in `asset_contract.py`:

```python
@dataclass(frozen=True)
class SmiTedAssetContract:
    parent_repository_historical: str
    parent_repository_canonical: str
    parent_revision: str
    parent_filename: str
    parent_sha256: str
    architecture: Mapping[str, int]
    vocab_name: str
    vocab_size: int
    vocab_sha256: str
    prefix_map: Mapping[str, str]
    allowed_excluded_prefixes: tuple[str, ...]
    tensor_count: int
    tensor_bytes: int
    parameter_bytes: int
    buffer_bytes: int


SMI_TED_LIGHT_CONTRACT = SmiTedAssetContract(
    parent_repository_historical="ibm/materials.smi-ted",
    parent_repository_canonical="ibm-research/materials.smi-ted",
    parent_revision="414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
    parent_filename="smi-ted-Light_40.pt",
    parent_sha256="baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
    architecture={"n_layer": 12, "n_head": 12, "n_embd": 768, "max_len": 202, "num_feats": 32},
    vocab_name="smi-ted-regex-v1",
    vocab_size=2393,
    vocab_sha256="8576b60e838336837f9e894457ef144f440c4d31bc8ceda2315fb5b28a6dfd95",
    prefix_map={
        "encoder.tok_emb.": "encoder.tok_emb.",
        "encoder.blocks.": "encoder.blocks.",
        "decoder.autoencoder.encoder.": "projector.",
    },
    allowed_excluded_prefixes=(
        "encoder.lang_model.",
        "decoder.autoencoder.decoder.",
        "decoder.lang_model.",
    ),
    tensor_count=224,
    tensor_bytes=656_641_536,
    parameter_bytes=656_541_696,
    buffer_bytes=99_840,
)
```

Expose mappings through immutable views or copy them on access so tests cannot mutate the process-wide contract.

- [ ] **Step 4: Implement only the embedding graph**

In `inference.py`, construct `SmiTedInferenceEncoder` without `lang_model` and
construct `SmiTedProjector` with the same `fc1`, `ln_f`, and bias-free `lat`
layers as the existing autoencoder encoder. Do not inherit its global
CUDA-availability behavior: both new modules operate on the device selected by
`.to(device)`. Port only the current tokenization/embedding/invalid-SMILES
behavior:

```python
class SmiTedInferenceModel(nn.Module):
    def __init__(self, tokenizer, config=None):
        super().__init__()
        self.config = dict(config or production_config())
        self.tokenizer = tokenizer
        self.padding_idx = tokenizer.get_padding_idx()
        self.max_len = self.config["max_len"]
        self.n_embd = self.config["n_embd"]
        self.encoder = SmiTedInferenceEncoder(self.config, len(tokenizer.vocab))
        self.projector = SmiTedProjector(
            self.max_len * self.n_embd, self.n_embd
        )

    def extract_embeddings(self, smiles):
        idx, mask = self.tokenize(smiles)
        token_embeddings = self.encoder(idx, mask)
        embedding = self.projector(
            token_embeddings.reshape(-1, self.max_len * self.n_embd)
        )
        return idx, token_embeddings, embedding
```

`production_config()` also supplies the state-free construction value
`d_dropout=0.2`. Do not copy `forward`, `decode`, `extract_all`, RNG
restoration, `LangLayer`, `MoLDecoder`, or `Net`. Move token tensors to the
actual model device instead of consulting global CUDA availability.

- [ ] **Step 5: Add semantic-equivalence tests using tiny deterministic weights**

Build a tiny full `Smi_ted` and inference model, map the three approved key prefixes, set both to eval mode, and require exact embedding equality, preserved `(N, n_embd)` shape, and NaN reinsertion for invalid SMILES:

```python
assert torch.equal(
    full_model.encode(["CCO", "c1ccccc1"], return_torch=True),
    inference_model.encode(["CCO", "c1ccccc1"], return_torch=True),
)
assert torch.isnan(inference_model.encode(["invalid"], return_torch=True)).all()
```

- [ ] **Step 6: Run focused tests**

Run: `conda run -n tsac_env python -m pytest tests/test_smi_ted_inference.py -v`

Expected: all inference contract, structure, and tiny-equivalence tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/tennetsac/smi_ted_light/asset_contract.py src/tennetsac/smi_ted_light/inference.py tests/test_smi_ted_inference.py
git commit -m "feat: add inference-only SMI-TED model"
```

---

### Task 2: Add strict safetensors validation and loading

**Files:**
- Modify: `src/tennetsac/smi_ted_light/asset_contract.py`
- Modify: `src/tennetsac/smi_ted_light/inference.py`
- Modify: `tests/test_smi_ted_inference.py`

**Interfaces:**
- Consumes: `SmiTedInferenceModel` and one schema-v2 SMI-TED manifest entry.
- Produces: `expected_safetensors_metadata(asset_entry) -> dict[str, str]` and `load_smi_ted_inference(checkpoint_path, vocab_path, asset_entry) -> SmiTedInferenceModel`.

- [ ] **Step 1: Write failing strict-loader tests**

Use `safetensors.torch.save_file` to create tiny local files and assert the loader rejects each mutation before model use:

```python
@pytest.mark.parametrize("mutation, expected", [
    (lambda state, metadata: state.pop(next(iter(state))), "missing keys"),
    (lambda state, metadata: state.update({"unexpected.weight": torch.zeros(1)}), "unexpected keys"),
    (lambda state, metadata: metadata.update(format_version="2"), "format_version"),
    (lambda state, metadata: metadata.update(vocab_size="999"), "vocab_size"),
])
def test_strict_loader_rejects_incompatible_asset(
    tmp_path, tiny_model_entry, tiny_model_factory, tiny_vocab_path,
    mutation, expected
):
    model = tiny_model_factory()
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    metadata = expected_safetensors_metadata(tiny_model_entry)
    mutation(state, metadata)
    checkpoint = tmp_path / "invalid.safetensors"
    save_file(state, checkpoint, metadata=metadata)
    with pytest.raises(ValueError, match=expected):
        load_smi_ted_inference(
            checkpoint, tiny_vocab_path, tiny_model_entry
        )
```

Also monkeypatch `torch.load` to raise if called and prove a valid safetensors file loads without invoking it.
Assert the metadata contains no digest for the safetensors file itself; that
digest belongs in the manifest and `SHA256SUMS`, avoiding self-reference.

- [ ] **Step 2: Run the new tests and verify the loader is missing**

Run: `conda run -n tsac_env python -m pytest tests/test_smi_ted_inference.py -k 'loader or metadata' -v`

Expected: failure because `load_smi_ted_inference` and metadata validation do not exist.

- [ ] **Step 3: Implement exact metadata construction**

Return strings for the approved fields, including JSON-encoded prefix mappings with sorted keys:

```python
def expected_safetensors_metadata(asset_entry):
    return {
        "format": "tennetsac-smi-ted-inference",
        "format_version": "1",
        "upstream_repository": SMI_TED_LIGHT_CONTRACT.parent_repository_canonical,
        "upstream_revision": SMI_TED_LIGHT_CONTRACT.parent_revision,
        "upstream_filename": SMI_TED_LIGHT_CONTRACT.parent_filename,
        "upstream_sha256": SMI_TED_LIGHT_CONTRACT.parent_sha256,
        "pruning_rule_version": "1",
        "included_prefixes": json.dumps(list(SMI_TED_LIGHT_CONTRACT.prefix_map), separators=(",", ":")),
        "key_mapping": json.dumps(dict(SMI_TED_LIGHT_CONTRACT.prefix_map), sort_keys=True, separators=(",", ":")),
        "n_layer": str(asset_entry["architecture"]["n_layer"]),
        "n_head": str(asset_entry["architecture"]["n_head"]),
        "n_embd": str(asset_entry["architecture"]["n_embd"]),
        "max_len": str(asset_entry["architecture"]["max_len"]),
        "num_feats": str(asset_entry["architecture"]["num_feats"]),
        "vocab_name": SMI_TED_LIGHT_CONTRACT.vocab_name,
        "vocab_size": str(asset_entry["vocab_size"]),
        "vocab_sha256": SMI_TED_LIGHT_CONTRACT.vocab_sha256,
        "state_tensor_count": str(asset_entry["state_tensor_count"]),
        "state_tensor_bytes": str(asset_entry["state_tensor_bytes"]),
    }
```

- [ ] **Step 4: Implement validate-before-load behavior**

Open with
`safe_open(checkpoint_path, framework="pt", device="cpu")`, compare the exact
metadata dictionary and key set, load tensors with
`safetensors.torch.load_file`, then compare every expected shape/dtype and total
bytes before `load_state_dict(state, strict=True)`. Hash verification remains
the asset manager's responsibility; this loader validates the model-format
contract.

- [ ] **Step 5: Run focused and tokenizer regression tests**

Run: `conda run -n tsac_env python -m pytest tests/test_smi_ted_inference.py tests/test_smi_ted_tokenizer.py -v`

Expected: all tests pass and the no-`torch.load` assertion is exercised.

- [ ] **Step 6: Commit**

```bash
git add src/tennetsac/smi_ted_light/asset_contract.py src/tennetsac/smi_ted_light/inference.py tests/test_smi_ted_inference.py
git commit -m "feat: strictly load SMI-TED safetensors"
```

---

### Task 3: Build the reproducible offline converter

**Files:**
- Create: `scripts/build_smi_ted_inference_asset.py`
- Create: `tests/test_smi_ted_converter.py`

**Interfaces:**
- Consumes: an explicit verified parent `.pt` path and a nonexistent output directory.
- Produces: `select_inference_state(source_state) -> dict[str, Tensor]`, `build_asset(parent_path, output_dir, converter_commit) -> BuildResult`, the safetensors file, `SHA256SUMS`, `IBM-materials-APACHE-2.0.txt`, and `SMI_TED_INFERENCE_PROVENANCE.md`.

- [ ] **Step 1: Write failing prefix-selection tests**

Cover all three included prefixes, all three recognized excluded prefixes, key normalization, unknown source keys, duplicate normalized keys, non-tensors, missing included prefixes, and parent structure:

```python
def test_select_inference_state_maps_only_required_tensors():
    source = {
        "encoder.tok_emb.weight": torch.ones(2, 2),
        "encoder.blocks.layers.0.weight": torch.ones(2, 2),
        "decoder.autoencoder.encoder.fc1.weight": torch.ones(2, 2),
        "encoder.lang_model.head.weight": torch.ones(2, 2),
        "decoder.autoencoder.decoder.rec.weight": torch.ones(2, 2),
        "decoder.lang_model.head.weight": torch.ones(2, 2),
    }
    selected = select_inference_state(source)
    assert set(selected) == {
        "encoder.tok_emb.weight",
        "encoder.blocks.layers.0.weight",
        "projector.fc1.weight",
    }
```

- [ ] **Step 2: Run the converter tests and confirm the script is absent**

Run: `conda run -n tsac_env python -m pytest tests/test_smi_ted_converter.py -v`

Expected: collection fails because `scripts.build_smi_ted_inference_asset` does not exist.

- [ ] **Step 3: Implement safe parent verification and selection**

Hash the file in 1 MiB chunks before deserializing. Require the exact digest by default and call:

```python
checkpoint = torch.load(
    parent_path,
    map_location=torch.device("cpu"),
    weights_only=True,
    mmap=True,
)
```

Require exact top-level keys needed by conversion, validate the five architecture values in `hparams`, accept only the six known source-prefix groups, and normalize only `decoder.autoencoder.encoder.` to `projector.`.

- [ ] **Step 4: Write failing no-overwrite and release-material tests**

Assert a pre-existing output directory fails before any file changes. Exercise
the release-file writer with controlled synthetic tensors and metadata; assert
`SHA256SUMS` has one lowercase digest for the safetensors, Apache license, and
provenance files and does not attempt to hash itself. The copied Apache file
must match the repository digest, and provenance must include both upstream
repository IDs, revision, parent digest, conversion command, prefix lists, and
converter commit. The production-only reload/count gate is exercised with the
real checkpoint in Task 4.

- [ ] **Step 5: Implement atomic artifact generation**

Create the output directory exactly once, save CPU-contiguous float32 tensors with `safetensors.torch.save_file`, reload through `load_smi_ted_inference`, and remove the newly created output directory on any failed build. Return a frozen result:

```python
@dataclass(frozen=True)
class BuildResult:
    artifact_path: Path
    artifact_size: int
    artifact_sha256: str
    tensor_count: int
    tensor_bytes: int
```

The CLI requires `--parent`, `--output-dir`, and `--converter-commit`; it performs no network operation and prints the result fields as stable `name=value` lines.

- [ ] **Step 6: Run converter tests**

Run: `conda run -n tsac_env python -m pytest tests/test_smi_ted_converter.py -v`

Expected: every selection, safety, checksum, and provenance test passes.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_smi_ted_inference_asset.py tests/test_smi_ted_converter.py
git commit -m "feat: convert SMI-TED to inference asset"
```

---

### Task 4: Generate the real candidate, prove parity, and pin manifest v2

**Files:**
- Modify: `src/tennetsac/_manifest_schema.py`
- Modify: `src/tennetsac/model_manifest.json`
- Modify: `tests/test_model_manifest.py`
- Modify: `tests/test_distribution.py`
- Create: `tests/integration/test_smi_ted_asset.py`

**Interfaces:**
- Consumes: the local parent checkpoint and converter from Task 3.
- Produces: one locally retained release-candidate directory, a concrete schema-v2 manifest record, and real-model parity tests.

- [ ] **Step 1: Build the real candidate in a new temporary directory**

Confirm the parent exists, confirm the dedicated candidate path does not exist,
and record the current converter commit before passing that new path to the
converter:

```bash
test -f /Users/yueyang/.cache/huggingface/hub/models--ibm--materials.smi-ted/snapshots/414c3ea0a8603ef49d1c5bb3db336e09877c01ce/smi-ted-Light_40.pt
test ! -e /private/tmp/tennetsac-smi-ted-light-v1-candidate-20260907
CONVERTER_COMMIT="$(git rev-parse HEAD)"
conda run -n tsac_env python scripts/build_smi_ted_inference_asset.py \
  --parent /Users/yueyang/.cache/huggingface/hub/models--ibm--materials.smi-ted/snapshots/414c3ea0a8603ef49d1c5bb3db336e09877c01ce/smi-ted-Light_40.pt \
  --output-dir /private/tmp/tennetsac-smi-ted-light-v1-candidate-20260907 \
  --converter-commit "$CONVERTER_COMMIT"
```

Expected: the command prints an approximately 657 MB path, exact SHA-256,
`tensor_count=224`, and `tensor_bytes=656641536`. Preserve
`/private/tmp/tennetsac-smi-ted-light-v1-candidate-20260907` for Tasks 4, 8,
and 9; do not copy it into the repository. If the preflight path check fails,
stop and inspect the existing directory instead of deleting or overwriting it.

- [ ] **Step 2: Write the failing complete-manifest test with the real digest**

Read the artifact's digest from the generated `SHA256SUMS`; use that exact lowercase 64-hex value in `EXPECTED_EXTERNAL_MODELS`. Define the SMI-TED entry with exact keys and concrete values:

```python
{
    "name": "smi-ted-light",
    "distribution": "external",
    "format": "safetensors",
    "format_version": 1,
    "repository": "yueyue2299/TeNNet-SAC",
    "release_tag": "model-smi-ted-light-v1",
    "url": "https://github.com/yueyue2299/TeNNet-SAC/releases/download/model-smi-ted-light-v1/smi-ted-light-inference-v1.safetensors",
    "filename": "smi-ted-light-inference-v1.safetensors",
    "sha256": artifact_digest_from_SHA256SUMS,
    "state_tensor_count": 224,
    "state_tensor_bytes": 656641536,
    "vocab_size": 2393,
    "architecture": {"n_layer": 12, "n_head": 12, "n_embd": 768, "max_len": 202, "num_feats": 32},
    "parent": {
        "historical_repository": "ibm/materials.smi-ted",
        "canonical_repository": "ibm-research/materials.smi-ted",
        "revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
        "filename": "smi-ted-Light_40.pt",
        "sha256": "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
    },
    "pruning": {
        "rule_version": 1,
        "included_prefixes": ["encoder.tok_emb.", "encoder.blocks.", "decoder.autoencoder.encoder."],
    },
    "legacy_override_env": "TENNETSAC_SMI_TED_CHECKPOINT",
}
```

Run: `conda run -n tsac_env python -m pytest tests/test_model_manifest.py -v`

Expected: failure because the shipped manifest and validator still use schema version 1.

- [ ] **Step 3: Implement exact schema version 2**

Update `_manifest_schema.py` so schema version must be integer `2`. Keep the ChemBERTa2 entry unchanged; validate the SMI-TED nested record with exact allowed/required key sets, immutable HTTPS GitHub URL, filename/tag agreement, lowercase SHA-256 values, positive exact integer fields, architecture values, parent values, prefix order/content, and override variable name.

- [ ] **Step 4: Update the shipped manifest with the generated digest**

Change only `schema_version` and the `smi-ted-light` entry in `model_manifest.json`; retain `bundle_version: "1.0.0"`, all thirteen bundled checkpoint records, ChemBERTa2, and tokenizer records byte-for-value. Never enter a guessed digest: the value must match both the generated safetensors file and `SHA256SUMS`.

- [ ] **Step 5: Write the real parity integration test**

Make the test require both `TENNETSAC_SMI_TED_PARENT_CHECKPOINT` and
`TENNETSAC_SMI_TED_INFERENCE_CHECKPOINT`, then compare a representative corpus
including ethanol, benzene, acetic acid, halogens, stereochemistry, and one
invalid SMILES. Compute and clone the legacy embeddings first, then delete the
full model and run garbage collection before loading the inference asset so the
test does not retain both large models simultaneously:

```python
assert torch.equal(
    legacy.encode(valid_smiles, return_torch=True),
    inference.encode(valid_smiles, return_torch=True),
)
assert torch.isnan(inference.encode(["not-a-smiles"], return_torch=True)).all()
assert len(inference.state_dict()) == 224
assert sum(t.numel() * t.element_size() for t in inference.state_dict().values()) == 656_641_536
```

Also assert the manifest digest equals the actual candidate digest, metadata is exact, and forbidden state prefixes/attributes are absent.

- [ ] **Step 6: Run manifest, distribution-fixture, and real parity tests**

Run the unit tests:

```bash
conda run -n tsac_env python -m pytest tests/test_model_manifest.py tests/test_distribution.py -v
```

Run the real test with the generated candidate path:

```bash
TENNETSAC_SMI_TED_PARENT_CHECKPOINT=/Users/yueyang/.cache/huggingface/hub/models--ibm--materials.smi-ted/snapshots/414c3ea0a8603ef49d1c5bb3db336e09877c01ce/smi-ted-Light_40.pt \
TENNETSAC_SMI_TED_INFERENCE_CHECKPOINT=/private/tmp/tennetsac-smi-ted-light-v1-candidate-20260907/smi-ted-light-inference-v1.safetensors \
conda run -n tsac_env python -m pytest tests/integration/test_smi_ted_asset.py -v
```

Expected: all tests pass with bitwise equality and the exact counts.

- [ ] **Step 7: Commit code and manifest, not the large candidate**

```bash
git add src/tennetsac/_manifest_schema.py src/tennetsac/model_manifest.json tests/test_model_manifest.py tests/test_distribution.py tests/integration/test_smi_ted_asset.py
git commit -m "feat: pin SMI-TED inference asset manifest"
```

---

### Task 5: Implement locked cache resolution, HTTPS download, and CLI

**Files:**
- Create: `src/tennetsac/_model_assets.py`
- Create: `src/tennetsac/model_assets.py`
- Create: `tests/test_model_assets.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_package_metadata.py`

**Interfaces:**
- Consumes: `external_model("smi-ted-light")` schema-v2 record.
- Produces:
  `ModelAssetError`, `asset_cache_path(asset_entry) -> Path`,
  `ResolvedModelAsset(path, sha256, format, is_legacy, manifest_entry)`,
  `resolve_model_asset(name="smi-ted-light", allow_download=True)`,
  `verify_model_asset(name="smi-ted-light")`, and
  `python -m tennetsac.model_assets {download,verify} smi-ted-light`.

- [ ] **Step 1: Write failing path, override, and offline tests**

Assert cache-root precedence, the exact default relative path, an explicit override winning without modification, and no downloader call in offline mode:

```python
def test_cache_path_uses_versioned_model_directory(monkeypatch, tmp_path, smi_entry):
    monkeypatch.setenv("TENNETSAC_CACHE_DIR", str(tmp_path))
    assert asset_cache_path(smi_entry) == (
        tmp_path / "models" / "smi-ted-light" / "v1"
        / "smi-ted-light-inference-v1.safetensors"
    )


def test_offline_cache_miss_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("TENNETSAC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TENNETSAC_OFFLINE", "1")
    with pytest.raises(ModelAssetError, match="python -m tennetsac.model_assets download"):
        resolve_model_asset()
```

- [ ] **Step 2: Run focused tests and verify the module is absent**

Run: `conda run -n tsac_env python -m pytest tests/test_model_assets.py -v`

Expected: collection fails because `_model_assets` does not exist.

- [ ] **Step 3: Implement torch-free paths, hashing, and per-process verification cache**

Add direct dependencies `platformdirs` and `filelock` to `pyproject.toml`. Key the verified-hash cache by `(resolved_path, size, mtime_ns, expected_sha256)` so replacing a path invalidates the cached result. Provide an explicit private cache-clear hook for tests.

- [ ] **Step 4: Write failing download-integrity tests**

Use a fake HTTPS response object with `geturl()`, `read()`, and headers. Cover
valid streaming, initial/final non-HTTPS URL rejection, hash mismatch,
interrupted read, write/fsync failure, lock timeout naming the lock and target,
lock recheck, corrupt cache replacement, offline corrupt-cache removal, and a
user-owned override that is rejected but never removed. Prove that a false or
missing `Content-Length` never substitutes for SHA-256 verification. Each
raised error must include the model name, release tag, URL or path, expected
digest, recovery command, and the original exception as its cause where one
exists.

- [ ] **Step 5: Implement locked atomic download**

Use `FileLock` adjacent to the final path. After locking, recheck the final file;
stream to a same-directory name containing `.part`, process ID, and UUID;
update SHA-256 per chunk; flush and `os.fsync`; and call `os.replace` only after
exact verification. A `finally` block removes only that invocation's partial
path. Preserve underlying exceptions with
`raise ModelAssetError(message) from error`.

- [ ] **Step 6: Implement format-aware override resolution**

Return `format="safetensors"` only for an exact derived digest and `format="pytorch"`, `is_legacy=True` only for an exact parent digest. Reject other suffixes and hashes. Do not search Hugging Face caches or import `huggingface_hub`.

- [ ] **Step 7: Write and implement CLI behavior**

Tests call `main(["download", "smi-ted-light"])` and `main(["verify", "smi-ted-light"])`, asserting exit code `0` plus path/digest output on success and nonzero plus actionable stderr on failure. `verify` always calls resolution with downloads disabled. Implement:

```python
def main(argv=None) -> int:
    args = parser().parse_args(argv)
    resolved = (
        resolve_model_asset(args.model, allow_download=True)
        if args.command == "download"
        else verify_model_asset(args.model)
    )
    print(f"path={resolved.path}")
    print(f"sha256={resolved.sha256}")
    return 0
```

- [ ] **Step 8: Verify import isolation and cache/CLI tests**

Extend the plain-import subprocess assertion to include `_model_assets`, `model_assets`, `platformdirs`, `filelock`, and `safetensors` in the modules that must remain unloaded.

Run: `conda run -n tsac_env python -m pytest tests/test_model_assets.py tests/test_package_metadata.py -v`

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/tennetsac/_model_assets.py src/tennetsac/model_assets.py tests/test_model_assets.py tests/test_package_metadata.py
git commit -m "feat: manage external model assets"
```

---

### Task 6: Wire default inference loading and exact legacy compatibility

**Files:**
- Modify: `src/tennetsac/smi_ted_light/load.py`
- Modify: `src/tennetsac/utils/embedding.py`
- Modify: `src/tennetsac/runtime.py`
- Modify: `tests/test_smi_ted_loading.py`
- Modify: `tests/test_embeddings.py`
- Modify: `tests/test_runtime.py`
- Modify: `tests/test_full_model_guards.py`

**Interfaces:**
- Consumes: `ResolvedModelAsset` and both model loaders.
- Produces: `load_legacy_smi_ted(checkpoint_path, vocab_path, expected_sha256)`, `load_smi_ted(checkpoint_path, vocab_path, asset_entry)`, and `SMITEDEmbedder(checkpoint_path, vocab_path, asset_entry, device="cpu")`.

- [ ] **Step 1: Replace obsolete Hugging Face loader tests with failing dispatch tests**

Delete tests expecting `hf_hub_download`. Add tests proving `.safetensors` calls only `load_smi_ted_inference`, exact `.pt` emits `FutureWarning` and calls only the legacy loader, and unknown formats fail:

```python
with pytest.warns(FutureWarning, match="next major"):
    model = load_smi_ted(
        checkpoint_path=legacy_path,
        vocab_path=vocab_path,
        asset_entry=smi_entry,
    )
```

Monkeypatch `torch.load` and assert wrong legacy bytes are rejected before deserialization.

- [ ] **Step 2: Run loading tests and confirm current auto-download behavior fails them**

Run: `conda run -n tsac_env python -m pytest tests/test_smi_ted_loading.py tests/test_embeddings.py tests/test_runtime.py -v`

Expected: failures show the old folder/repository/filename interface and Hugging Face fallback.

- [ ] **Step 3: Make `load.py` legacy-only and restricted**

Remove the top-level `huggingface_hub` import and all download logic. Hash the
explicit `.pt` before calling
`torch.load(checkpoint_path, weights_only=True, mmap=True, map_location="cpu")`;
require the exact parent SHA, emit one `FutureWarning`, instantiate the existing
full model, load it, set eval mode, and return it. Do not weaken its checkpoint
key handling beyond the temporary compatibility contract.

- [ ] **Step 4: Add one format dispatcher and simplify the embedder**

Add the dispatcher beside the legacy loader in `smi_ted_light/load.py`, then
have `utils/embedding.py` call only that interface:

```python
def load_smi_ted(checkpoint_path, vocab_path, asset_entry):
    if checkpoint_path.suffix == ".safetensors":
        return load_smi_ted_inference(checkpoint_path, vocab_path, asset_entry)
    if checkpoint_path.suffix == ".pt":
        return load_legacy_smi_ted(
            checkpoint_path,
            vocab_path,
            asset_entry["parent"]["sha256"],
        )
    raise ValueError(f"Unsupported SMI-TED checkpoint format: {checkpoint_path.suffix}")
```

`SMITEDEmbedder` passes explicit paths and retains its CPU-returning callable behavior.

- [ ] **Step 5: Wire runtime resolution once**

Resolve `smi-ted-light` through `_model_assets.resolve_model_asset` inside `_build_runtime`, materialize only the packaged vocab with `_smi_ted_vocab_dir`, and pass the resolved path plus manifest entry to `SMITEDEmbedder`. Keep `get_runtime()` as the single reuse cache. Wrap errors with `Failed to initialize SMI-TED embedder` while preserving the asset error as `__cause__`.

- [ ] **Step 6: Prove no Hugging Face fallback or duplicate resolution**

Tests must assert one resolver call per runtime build, explicit override propagation, offline failure context, and absence of `huggingface_hub` from the SMI-TED loading module. Keep the ChemBERTa2 cache guards unchanged.

- [ ] **Step 7: Run runtime-focused regressions**

Run:

```bash
conda run -n tsac_env python -m pytest \
  tests/test_smi_ted_loading.py tests/test_embeddings.py tests/test_runtime.py \
  tests/test_full_model_guards.py tests/test_package_metadata.py -v
```

Expected: all tests pass; no test expects a Hugging Face SMI-TED download.

- [ ] **Step 8: Commit**

```bash
git add src/tennetsac/smi_ted_light/load.py src/tennetsac/utils/embedding.py src/tennetsac/runtime.py tests/test_smi_ted_loading.py tests/test_embeddings.py tests/test_runtime.py tests/test_full_model_guards.py
git commit -m "feat: load pruned SMI-TED by default"
```

---

### Task 7: Document provenance and enforce package boundaries

**Files:**
- Modify: `README.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `src/tennetsac/assets/README.md`
- Modify: `MANIFEST.in`
- Modify: `scripts/verify_distribution.py`
- Modify: `tests/test_documentation.py`
- Modify: `tests/test_licensing.py`
- Modify: `tests/test_distribution.py`

**Interfaces:**
- Consumes: the final CLI, environment variables, release identity, and derived-asset provenance.
- Produces: user-visible prefetch/offline/override instructions and archive checks that reject external model weights.

- [ ] **Step 1: Write failing documentation and licensing assertions**

Require README text for both CLI commands, all three environment variables,
cache behavior, immutable release tag, first-call download, and the legacy
warning. Require it to state that a defective v1 is replaced by a new model
release such as `model-smi-ted-light-v2` plus a package patch, never by moving
or replacing v1. Require the notice to contain the canonical parent repository,
revision, parent digest, derived asset name, pruning statement, 224 tensors,
Apache-2.0, and no-endorsement wording.

- [ ] **Step 2: Write failing archive-boundary tests**

Add wheel/sdist cases containing either `smi-ted-Light_40.pt` or `smi-ted-light-inference-v1.safetensors` anywhere in the archive, and require `verify_archive()` to reject both. Assert the clean fixture still passes with schema v2.

- [ ] **Step 3: Run the tests and observe missing documentation/boundary behavior**

Run: `conda run -n tsac_env python -m pytest tests/test_documentation.py tests/test_licensing.py tests/test_distribution.py -v`

Expected: the new facts and safetensors rejection tests fail.

- [ ] **Step 4: Update user and attribution documentation**

Document:

```text
python -m tennetsac.model_assets download smi-ted-light
python -m tennetsac.model_assets verify smi-ted-light
TENNETSAC_CACHE_DIR
TENNETSAC_SMI_TED_CHECKPOINT
TENNETSAC_OFFLINE=1
```

State that the default derivative is distributed by TeNNet-SAC under the retained upstream Apache-2.0 terms, is not an IBM-published checkpoint, and does not imply IBM endorsement. Keep the future ChemBERTa2 directory description accurate.

- [ ] **Step 5: Harden source and binary distribution exclusions**

Add `global-exclude *.pt *.safetensors` to `MANIFEST.in`. In `verify_distribution.py`, reject those suffixes as external weights while retaining the exact thirteen `.ckpt` package artifacts. Do not reject the safetensors Python dependency or textual manifest references.

- [ ] **Step 6: Run documentation and archive tests**

Run: `conda run -n tsac_env python -m pytest tests/test_documentation.py tests/test_licensing.py tests/test_distribution.py -v`

Expected: all tests pass.

- [ ] **Step 7: Build and inspect actual distributions**

Run:

```bash
conda run -n tsac_env python -m build
conda run -n tsac_env python -m twine check dist/*
conda run -n tsac_env python scripts/verify_distribution.py dist/*
```

Expected: wheel/sdist pass and contain no `.pt` or `.safetensors` model files.

- [ ] **Step 8: Commit**

```bash
git add README.md THIRD_PARTY_NOTICES.md src/tennetsac/assets/README.md MANIFEST.in scripts/verify_distribution.py tests/test_documentation.py tests/test_licensing.py tests/test_distribution.py
git commit -m "docs: document SMI-TED model asset"
```

---

### Task 8: Review the branch and stop at the external release authorization gate

**Files:**
- Modify only files required by concrete review findings.
- Preserve: the local release-candidate directory from Task 4.

**Interfaces:**
- Consumes: Tasks 1-7 and the approved design.
- Produces: a reviewed, locally verified branch plus an exact proposed `gh release create` command; no external mutation yet.

- [ ] **Step 1: Run the complete offline suite**

Run: `conda run -n tsac_env python -m pytest -m "not integration" -v`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run the real asset and existing golden integrations locally**

Set the exact parent and generated candidate paths. Keep Hugging Face/Transformers offline for cached ChemBERTa2:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TENNETSAC_SMI_TED_PARENT_CHECKPOINT=/Users/yueyang/.cache/huggingface/hub/models--ibm--materials.smi-ted/snapshots/414c3ea0a8603ef49d1c5bb3db336e09877c01ce/smi-ted-Light_40.pt \
TENNETSAC_SMI_TED_INFERENCE_CHECKPOINT=/private/tmp/tennetsac-smi-ted-light-v1-candidate-20260907/smi-ted-light-inference-v1.safetensors \
TENNETSAC_SMI_TED_CHECKPOINT=/private/tmp/tennetsac-smi-ted-light-v1-candidate-20260907/smi-ted-light-inference-v1.safetensors \
conda run -n tsac_env python -m pytest tests/integration/test_smi_ted_asset.py tests/integration/test_full_model.py -v
```

Expected: bitwise SMI-TED parity and existing PyPI 0.1.10 golden tolerances pass.

- [ ] **Step 3: Perform task-scoped specification review**

Compare every acceptance criterion in the design against code, tests, and fresh command output. Inspect `git diff 8980fd5...HEAD`, `git status`, candidate checksums, and artifact sizes. Fix only concrete findings and rerun their focused tests before committing each fix.

- [ ] **Step 4: Prepare but do not execute the release command**

After verifying the candidate directory contains exactly the four approved files, prepare this command with the exact Task 4 paths:

```bash
gh release create model-smi-ted-light-v1 \
  smi-ted-light-inference-v1.safetensors \
  SHA256SUMS \
  IBM-materials-APACHE-2.0.txt \
  SMI_TED_INFERENCE_PROVENANCE.md \
  --repo yueyue2299/TeNNet-SAC \
  --title "SMI-TED Light inference asset v1" \
  --notes-file SMI_TED_INFERENCE_PROVENANCE.md
```

Do not run it. Report the verified digest, bytes, converter commit, branch status, and exact upload set to the repository owner, then request explicit authorization to create the external Release.

---

### Task 9: Add the package-release guard, integrate the branch, and publish the authorized model release

**Files:**
- Modify: `.github/workflows/release-build.yml`
- Modify: `tests/test_workflows.py`
- Modify: `tests/integration/test_smi_ted_asset.py`
- Modify: `tests/integration/test_full_model.py`

**Interfaces:**
- Consumes: explicit owner authorization at Task 8, the exact local candidate, and the manifest URL/digest.
- Produces: a package-release workflow that refuses to proceed if the asset is
  unavailable or invalid, an integrated converter commit, an immutable public
  model release, and a clean full-prediction smoke.

- [ ] **Step 1: Write failing public-smoke workflow tests**

Require `.github/workflows/release-build.yml` to run before package building:

```text
python -m tennetsac.model_assets download smi-ted-light
python -m tennetsac.model_assets verify smi-ted-light
python -m pytest tests/integration/test_full_model.py -k public_asset -v
```

The step must override repository-wide SMI-TED and Hugging Face offline
variables only for this release smoke, use a clean `TENNETSAC_CACHE_DIR`, and
fail before wheel/sdist building or draft package release creation.

- [ ] **Step 2: Add public asset and full-prediction smoke tests**

Add a public-asset test that resolves from a clean cache without
`TENNETSAC_SMI_TED_CHECKPOINT`, compares its digest to the manifest, loads it,
and encodes `CCO` to finite shape `(1, 768)`. Add a separate public full-model
test that calls `profile("CCO")` and requires a 51-point finite sigma profile
plus positive area and volume. Gate both tests on
`TENNETSAC_PUBLIC_ASSET_SMOKE=1` so ordinary integration runs remain local and
offline.

- [ ] **Step 3: Update the workflow and run static tests**

Place the online smoke after unit tests and before wheel/sdist build. Set
`TENNETSAC_OFFLINE: "0"`, `HF_HUB_OFFLINE: "0"`,
`TRANSFORMERS_OFFLINE: "0"`, `TENNETSAC_PUBLIC_ASSET_SMOKE: "1"`, and a new
temporary cache only on that step.

Run: `conda run -n tsac_env python -m pytest tests/test_workflows.py -v`

Expected: all workflow ordering, environment, and command assertions pass.

- [ ] **Step 4: Commit the release guard**

```bash
git add .github/workflows/release-build.yml tests/test_workflows.py tests/integration/test_smi_ted_asset.py tests/integration/test_full_model.py
git commit -m "ci: verify public SMI-TED asset before release"
```

- [ ] **Step 5: Run final pre-integration verification**

Run:

```bash
conda run -n tsac_env python -m pytest -m "not integration" -v
conda run -n tsac_env python -m build
conda run -n tsac_env python -m twine check dist/*
conda run -n tsac_env python scripts/verify_distribution.py dist/*
git diff --check 8980fd5...HEAD
git status --short --branch
```

Expected: zero test/build/distribution failures, no whitespace errors, and no
large model file tracked.

- [ ] **Step 6: Obtain integration authority and make the converter commit reachable**

Use the branch-finishing workflow to present the verified branch to the owner.
Do not push or merge without explicit authorization. Before creating the model
release, confirm the exact converter commit recorded in provenance is reachable
in `yueyue2299/TeNNet-SAC` and that the final schema-v2 manifest is present in
the integrated repository state.

- [ ] **Step 7: Confirm separate model-release authorization and remote preconditions**

Immediately before the external mutation, verify the authorized
repository/tag/assets, confirm `model-smi-ted-light-v1` does not already exist,
and verify the candidate hashes again. If any target differs from Task 8's
reviewed upload set, stop and request renewed authorization.

- [ ] **Step 8: Create the model release exactly once**

Run the reviewed Task 8 `gh release create` command from
`/private/tmp/tennetsac-smi-ted-light-v1-candidate-20260907`. Never use
`--clobber`, never replace an asset, and never move the tag. Confirm all four
public download URLs and the public `SHA256SUMS` contents. Enable immutable
release protection through the repository's supported release settings after
publication; if the repository does not expose that capability, record the
limitation rather than attempting a replacement-based workaround.

- [ ] **Step 9: Run the real clean-download and full-prediction smoke**

Run with new SMI-TED and Hugging Face caches and no checkpoint override:

```bash
test -z "${TENNETSAC_SMI_TED_CHECKPOINT:-}"
test ! -e /private/tmp/tennetsac-public-smoke-20260907
TENNETSAC_OFFLINE=0 \
HF_HUB_OFFLINE=0 \
TRANSFORMERS_OFFLINE=0 \
TENNETSAC_PUBLIC_ASSET_SMOKE=1 \
TENNETSAC_CACHE_DIR=/private/tmp/tennetsac-public-smoke-20260907/tennetsac \
HF_HOME=/private/tmp/tennetsac-public-smoke-20260907/huggingface \
conda run -n tsac_env python -m pytest \
  tests/integration/test_smi_ted_asset.py tests/integration/test_full_model.py \
  -k public_asset -v
```

Expected: the public URL downloads and verifies, SMI-TED produces a finite
`(1, 768)` embedding, and a full `profile("CCO")` prediction succeeds from the
clean environment. Do not create package tag `v0.2.0`, draft a package
release, or publish PyPI without their own explicit authorization.
