# Shared-Trunk Gamma Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ten duplicated fine-tuned gamma models with one approximately 5.2 MB shared-trunk ensemble that returns population standard deviation by default and supports selecting members `"1"` through `"10"`.

**Architecture:** A `GammaEnsemble` owns one `GammaTrunk` and ten ordinary PyTorch final heads. An offline, deterministic converter proves the existing checkpoints share exactly the approved tensors and produces one strict safetensors bundle; runtime and distribution metadata then switch atomically to that bundle. Mean-only prediction differentiates the averaged Gibbs energy once, while statistics prediction obtains ten member gradients from one shared forward graph with a batched vector-Jacobian product.

**Tech Stack:** Python 3.10-3.12, PyTorch 2.1.2+, safetensors, NumPy, pytest, setuptools, Twine.

**Spec:** `docs/superpowers/specs/2026-09-08-shared-trunk-ensemble-design.md`

## Global Constraints

- Work only on `codex/shared-trunk-ensemble`; do not mix ChemBERTa2 redistribution, package publication, or unrelated refactors into this branch.
- Preserve `Prf_to_Seg_Model`, `base.ckpt`, `geo.ckpt`, and `prf.ckpt` behavior and state-dictionary names.
- The only approved differing source keys are the six `model_final.{0,2,4}.{weight,bias}` tensors; all 33 other entries must compare bitwise equal across ten sources.
- Preserve tensor values and dtypes exactly; do not cast during conversion.
- The bundle path is exactly `src/tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors` and must contain exactly 93 tensors totaling exactly 5,202,560 tensor bytes.
- Keep manifest schema version 2 and change bundle version from `1.0.0` to `2.0.0` only when the real bundle is integrated.
- `return_std=True` is the new public default; population standard deviation uses `ddof=0` semantics.
- `return_std=True` is accepted only with `version="tuned"`; `"base"` and `"1"` through `"10"` require `return_std=False`.
- Numerical model results use `rtol=1e-5` and `atol=1e-6`; keys, values during conversion, file hashes, tensor counts, tensor bytes, and archive membership use exact checks.
- Do not overwrite conversion outputs or delete the only copy of any source checkpoint.
- Do not push, tag, create or edit a GitHub Release, or publish PyPI without separate explicit authorization after final review.

## File map

- Create `src/tennetsac/models/GammaEnsemble.py`: shared trunk, ten heads, mean/member/all-member gradient paths.
- Create `src/tennetsac/gamma_ensemble.py`: fixed bundle contract, digest/header validation, and strict safetensors loading.
- Create `scripts/_safetensors_canonical.py`: deterministic safetensors-header helper shared by both converters.
- Create `scripts/gamma_ensemble_sources.json`: immutable source checkpoint paths, source commit, bundle version, and hashes.
- Create `scripts/build_gamma_ensemble_asset.py`: no-overwrite deterministic converter.
- Create `scripts/generate_gamma_ensemble_golden.py`: reproducible pre-migration numerical fixture generator.
- Create `scripts/benchmark_gamma_ensemble.py`: controlled legacy/new latency and memory comparison.
- Create `tests/test_gamma_ensemble.py`: model mathematics, member selection, batched VJP, and one-trunk-call tests.
- Create `tests/test_gamma_ensemble_loader.py`: strict bundle validation and error tests.
- Create `tests/test_gamma_ensemble_converter.py`: source validation, mapping, no-overwrite, cleanup, and reproducibility tests.
- Create `tests/integration/test_gamma_ensemble_asset.py`: real ten-checkpoint conversion and parity gate.
- Create `tests/fixtures/gamma_ensemble_golden.json`: fixed pre-migration member, mean, and population-standard-deviation reference values.
- Modify `scripts/build_smi_ted_inference_asset.py` and `tests/test_smi_ted_converter.py`: import the shared canonical-header helper without changing SMI-TED output.
- Modify `src/tennetsac/models/Prf2Gamma.py`: reuse a private shared-forward helper while retaining the base model layout.
- Modify `src/tennetsac/utils/model_io.py`: load one strict ensemble instead of ten complete models.
- Modify `src/tennetsac/runtime.py`: materialize and cache the new bundled resource once.
- Modify `src/tennetsac/core.py` and `src/tennetsac/utils/property.py`: default statistics API and numbered-member dispatch.
- Modify `src/tennetsac/model_manifest.json`, `src/tennetsac/_manifest_schema.py`, and `tests/test_model_manifest.py`: bundle v2 inventory.
- Modify `pyproject.toml`, `scripts/verify_distribution.py`, and `tests/test_distribution.py`: package exactly one approved internal safetensors model.
- Create `tests/test_model_io.py` and modify `tests/test_runtime.py`, `tests/test_api_contract.py`, and relevant integration tests: strict model I/O, new private runtime shape, and public return contract.
- Create `docs/model-assets/gamma-ensemble-v1.md` and modify `README.md`, `src/tennetsac/assets/README.md`, and `tests/test_documentation.py`: behavior, project-owned model provenance, and breaking-change documentation.
- Delete `src/tennetsac/ckpt_files/fine-tuned/{1..10}.ckpt` only after verified copies and a verified replacement exist.

---

## Execution setup

At implementation time, invoke `superpowers:using-git-worktrees`, detect
whether the session is already isolated, and follow the user's existing
workspace preference. Confirm the execution branch starts at the committed
design/plan head and run this baseline before Task 1:

```bash
git status --short --branch
conda run -n tsac_env python -m pytest -m "not integration" -q
```

Expected baseline: a clean branch and the complete pre-change offline suite
passing. If it fails, stop and report the exact failures rather than mixing a
baseline repair into this feature.

---

### Task 1: Implement the shared-trunk ensemble mathematics

**Files:**
- Create: `src/tennetsac/models/GammaEnsemble.py`
- Modify: `src/tennetsac/models/Prf2Gamma.py`
- Create: `tests/test_gamma_ensemble.py`

**Interfaces:**
- Consumes: the existing layer dimensions and forward equations in `Prf_to_Seg_Model`.
- Produces: `GammaTrunk`, `GammaEnsemble(member_count: int = 10)`, and `GammaEnsemble.predict_segac(sigma, temperature, *, member_index=None, return_members=False) -> torch.Tensor`.
- `member_index` is zero-based internally; public one-based strings are mapped later in `core.py`.
- With neither option, the result shape is `(batch, 51)` and is the gradient of mean Gibbs energy. With `member_index`, the same shape contains one member. With `return_members=True`, shape is `(10, batch, 51)`.

- [ ] **Step 1: Add RED architecture and parity tests**

Create tests that copy one legacy model's 33 shared entries and each legacy
model's six final entries into a synthetic `GammaEnsemble`. Use three members
for focused tests and fixed seeds.

```python
SHARED_PREFIXES = (
    "model_sigma.", "bn_sig.", "temp_embedding.",
    "bn_t.", "model_combined.", "bn2.", "res_block.",
)

def _legacy_members(count=3):
    torch.manual_seed(4107)
    first = Prf_to_Seg_Model().eval()
    members = [first]
    for seed in range(4108, 4108 + count - 1):
        torch.manual_seed(seed)
        member = Prf_to_Seg_Model().eval()
        state = member.state_dict()
        for key, value in first.state_dict().items():
            if key.startswith(SHARED_PREFIXES):
                state[key].copy_(value)
        members.append(member)
    return members

def _ensemble_from_legacy(members):
    packed = {}
    for key, value in members[0].state_dict().items():
        if not key.startswith("model_final."):
            packed[f"trunk.{key}"] = value
    for index, member in enumerate(members):
        for key, value in member.state_dict().items():
            if key.startswith("model_final."):
                suffix = key.removeprefix("model_final.")
                packed[f"heads.{index}.{suffix}"] = value
    ensemble = GammaEnsemble(member_count=len(members))
    ensemble.load_state_dict(packed, strict=True)
    return ensemble

def test_mean_member_and_all_member_gradients_match_legacy_models():
    legacy = _legacy_members()
    ensemble = _ensemble_from_legacy(legacy).eval()
    sigma = torch.linspace(0.1, 1.1, 102).reshape(2, 51)
    temperature = torch.tensor([298.15, 315.0])
    expected = torch.stack([
        model(sigma.clone(), temperature)[1] for model in legacy
    ])

    torch.testing.assert_close(
        ensemble.predict_segac(sigma, temperature, return_members=True),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        ensemble.predict_segac(sigma, temperature),
        expected.mean(dim=0),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        ensemble.predict_segac(sigma, temperature, member_index=1),
        expected[1],
        rtol=1e-5,
        atol=1e-6,
    )
```

Also assert scalar temperature expansion, invalid member indices, mutual
exclusion of `member_index` and `return_members`, exact member result shape,
and rejection while the ensemble is in training mode.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
conda run -n tsac_env python -m pytest tests/test_gamma_ensemble.py -v
```

Expected: collection or import failure because `GammaEnsemble.py` and the
declared interfaces do not exist.

- [ ] **Step 3: Implement the architecture and the single-member path**

Create ordinary PyTorch modules with the exact established equations:

```python
class GammaTrunk(nn.Module):
    def __init__(self, sig_dim=51, temp_hidden_dim=16,
                 sig_hidden_dim=512, hidden_2_dim=256):
        super().__init__()
        self.model_sigma = nn.Sequential(
            nn.Linear(sig_dim, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, sig_hidden_dim), nn.GELU(),
        )
        self.bn_sig = nn.BatchNorm1d(sig_hidden_dim)
        self.temp_embedding = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, temp_hidden_dim), nn.ReLU(),
        )
        self.bn_t = nn.BatchNorm1d(temp_hidden_dim)
        self.model_combined = nn.Sequential(
            nn.Linear(sig_hidden_dim + temp_hidden_dim, 256), nn.GELU(),
            nn.Linear(256, hidden_2_dim), nn.GELU(),
        )
        self.bn2 = nn.BatchNorm1d(hidden_2_dim)
        self.res_block = ResidualBlock(
            in_features=hidden_2_dim,
            hidden_features=hidden_2_dim,
            activation=nn.GELU,
        )

    def forward(self, sigs, temperature):
        sum_sigs = sigs.sum(dim=1, keepdim=True)
        sigma_emb = self.bn_sig(self.model_sigma(sigs / sum_sigs))
        t = torch.as_tensor(temperature, dtype=sigs.dtype, device=sigs.device)
        if t.numel() == 1:
            t = t.reshape(1).expand(sigs.shape[0])
        elif t.numel() != sigs.shape[0]:
            raise ValueError("temperature must be scalar or match sigma batch size")
        t_emb = self.bn_t(self.temp_embedding((1.0 / t).reshape(-1, 1)))
        combined = self.bn2(self.model_combined(torch.cat([sigma_emb, t_emb], dim=-1)))
        return sum_sigs, self.res_block(combined)

class GammaEnsemble(nn.Module):
    def __init__(self, member_count=10):
        super().__init__()
        if type(member_count) is not int or member_count < 1:
            raise ValueError("member_count must be a positive integer")
        self.trunk = GammaTrunk()
        self.heads = nn.ModuleList(_new_final_head() for _ in range(member_count))

    def predict_segac(self, sigma, temperature, *, member_index=None,
                      return_members=False):
        if self.training:
            raise RuntimeError("GammaEnsemble prediction requires eval mode")
        if return_members and member_index is not None:
            raise ValueError("member_index and return_members are mutually exclusive")
        if member_index is not None and (
            type(member_index) is not int
            or not 0 <= member_index < len(self.heads)
        ):
            raise IndexError("member_index out of range")
        sigs = sigma.clone().detach().requires_grad_(True)
        sum_sigs, shared = self.trunk(sigs, temperature)
        if member_index is not None:
            gibbs = sum_sigs * self.heads[member_index](shared)
            return torch.autograd.grad(gibbs.sum(), sigs)[0]
        gibbs = torch.stack([sum_sigs * head(shared) for head in self.heads])
        # The remaining mean/all-member branches are added in Step 5.
```

Extract the final-network construction into a private
`_new_final_head(hidden_2_dim: int = 256) -> nn.Sequential` in
`Prf2Gamma.py`. It returns the exact existing sequence
`Linear(hidden_2_dim, 128)`, `GELU`, `Linear(128, 64)`, `GELU`, and
`Linear(64, 1)`. Import that factory into `GammaEnsemble.py`. Keep the base
forward calculation in place; do not nest the existing base modules under a
new prefix or alter its `state_dict()` keys.

- [ ] **Step 4: Run the single-member subset**

Run:

```bash
conda run -n tsac_env python -m pytest tests/test_gamma_ensemble.py -k 'member and not all_member' -v
```

Expected: single-member and validation tests pass; mean/all-member tests remain
RED until the next step.

- [ ] **Step 5: Implement mean and batched-VJP member gradients**

Use one reverse pass for mean and one batched VJP request for all members:

```python
if return_members:
    count, batch = gibbs.shape[:2]
    basis = torch.eye(count, dtype=gibbs.dtype, device=gibbs.device)
    basis = basis.reshape(count, count, 1, 1).expand(count, count, batch, 1)
    return torch.autograd.grad(
        gibbs,
        sigs,
        grad_outputs=basis,
        is_grads_batched=True,
    )[0]
return torch.autograd.grad(gibbs.mean(dim=0).sum(), sigs)[0]
```

Register a forward hook on `ensemble.trunk` in tests and require exactly one
call for mean, member, and all-member modes. Compare the batched VJP against
ten explicit legacy gradients.

- [ ] **Step 6: Run focused and base-model regression tests**

Run:

```bash
conda run -n tsac_env python -m pytest \
  tests/test_gamma_ensemble.py -v
```

Expected: all pass, including exact assertions that a new
`Prf_to_Seg_Model().state_dict()` still has the original 39 keys.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/tennetsac/models/GammaEnsemble.py \
  src/tennetsac/models/Prf2Gamma.py tests/test_gamma_ensemble.py
git commit -m "feat: add shared-trunk gamma ensemble"
```

---

### Task 2: Build the deterministic converter and immutable source contract

**Files:**
- Create: `scripts/_safetensors_canonical.py`
- Create: `scripts/gamma_ensemble_sources.json`
- Create: `scripts/build_gamma_ensemble_asset.py`
- Create: `tests/test_gamma_ensemble_converter.py`
- Modify: `scripts/build_smi_ted_inference_asset.py`
- Modify: `tests/test_smi_ted_converter.py`

**Interfaces:**
- Consumes: `GammaEnsemble.state_dict()` names from Task 1 and the exact ten source hashes in the spec.
- Produces: `build_bundle(source_dir: Path, output_path: Path, source_contract_path: Path) -> BundleReport` and a CLI accepting `SOURCE_DIR OUTPUT_PATH --source-contract PATH`.
- Produces shared `_canonicalize_safetensors_header(path: Path) -> None` without changing canonical SMI-TED bytes.

Define the converter result rather than returning an unstructured mapping:

```python
@dataclass(frozen=True)
class BundleReport:
    path: Path
    sha256: str
    tensor_count: int
    tensor_bytes: int
```

- [ ] **Step 1: Persist the exact source contract**

Create `scripts/gamma_ensemble_sources.json` with this exact content:

```json
{
  "format_version": 1,
  "source_bundle_version": "1.0.0",
  "source_commit": "7711c6bd3dff7e2d7cff00590062f9e1a0ca605a",
  "members": [
    {
      "member": 1,
      "path": "ckpt_files/fine-tuned/1.ckpt",
      "sha256": "134be46cce6839a7b08250750e4803c8198a4725d5973ca124db76dd488aedc1"
    },
    {
      "member": 2,
      "path": "ckpt_files/fine-tuned/2.ckpt",
      "sha256": "d763688bbe2898fc182ae6e9b39436a678ad892510c76b510c43c389e87d4c5b"
    },
    {
      "member": 3,
      "path": "ckpt_files/fine-tuned/3.ckpt",
      "sha256": "15232df78dbbce274818e00bdfff455a163fbb9ff8af1ef3ba4de73650519400"
    },
    {
      "member": 4,
      "path": "ckpt_files/fine-tuned/4.ckpt",
      "sha256": "bf01d2c6832b161ddf12d789dc886a3c3065a1810948d196f103a89c94f0daeb"
    },
    {
      "member": 5,
      "path": "ckpt_files/fine-tuned/5.ckpt",
      "sha256": "937a02283521bba254b917685f4db265c48617732f9c5bbfffb14eaaf1d38813"
    },
    {
      "member": 6,
      "path": "ckpt_files/fine-tuned/6.ckpt",
      "sha256": "7588bfe7a265752395373cf930ede2e721da2cb37fd7779a7b3b94d6c3590e79"
    },
    {
      "member": 7,
      "path": "ckpt_files/fine-tuned/7.ckpt",
      "sha256": "f4c03f8281b7ac56213e5e04123c208fc4f2030ca610684bd1cbd9aa59f7e16d"
    },
    {
      "member": 8,
      "path": "ckpt_files/fine-tuned/8.ckpt",
      "sha256": "0ff319a81281b4315638281283433590e6adf7b7ab17039919601747f736096d"
    },
    {
      "member": 9,
      "path": "ckpt_files/fine-tuned/9.ckpt",
      "sha256": "73c65639135717a3eaa7130c03ea6f85fa3f95ed1b5937b5b73611bd33378c70"
    },
    {
      "member": 10,
      "path": "ckpt_files/fine-tuned/10.ckpt",
      "sha256": "def4bb97274011564b33d55666bcc5d4d2bcae9c422cbf2c96529c7c364ebfcc"
    }
  ]
}
```

Tests require the ordered member sequence `list(range(1, 11))`, exact key
sets, lowercase hashes, and no additional fields. The converter resolves each
source as `source_dir / Path(contract_entry["path"]).name`, while retaining
the full logical path in diagnostics and provenance.

- [ ] **Step 2: Add RED converter tests**

Use temporary synthetic legacy checkpoints and a matching temporary source
contract. Cover successful 93-key mapping, a changed shared tensor, an
unexpected key, a missing member, an extra numbered `.ckpt`, wrong hash,
wrong dtype/shape, symlink source, existing output, interrupted save cleanup,
strict post-save reload, and two byte-identical conversions.

```python
def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _legacy_members(count):
    torch.manual_seed(4107)
    first = Prf_to_Seg_Model().eval()
    members = [first]
    for seed in range(4108, 4108 + count - 1):
        torch.manual_seed(seed)
        member = Prf_to_Seg_Model().eval()
        state = member.state_dict()
        for key, value in first.state_dict().items():
            if not key.startswith("model_final."):
                state[key].copy_(value)
        members.append(member)
    return members

def write_legacy_sources(tmp_path, count=10):
    sources = tmp_path / "sources"
    sources.mkdir()
    members = _legacy_members(count)
    records = []
    for number, model in enumerate(members, start=1):
        path = sources / f"{number}.ckpt"
        torch.save(model.state_dict(), path)
        records.append({
            "member": number,
            "path": f"ckpt_files/fine-tuned/{number}.ckpt",
            "sha256": _sha256(path),
        })
    contract = tmp_path / "sources.json"
    contract.write_text(json.dumps({
        "format_version": 1,
        "source_bundle_version": "1.0.0",
        "source_commit": "0" * 40,
        "members": records,
    }))
    return sources, contract

def rewrite_contract_hash(contract, changed_path):
    payload = json.loads(contract.read_text())
    number = int(changed_path.stem)
    payload["members"][number - 1]["sha256"] = _sha256(changed_path)
    contract.write_text(json.dumps(payload))

def test_converter_rejects_a_shared_tensor_that_differs(tmp_path):
    sources, contract = write_legacy_sources(tmp_path)
    changed = torch.load(sources / "2.ckpt", weights_only=True)
    changed["model_sigma.0.weight"] = changed["model_sigma.0.weight"].clone()
    changed["model_sigma.0.weight"][0, 0] += 1
    torch.save(changed, sources / "2.ckpt")
    rewrite_contract_hash(contract, sources / "2.ckpt")

    with pytest.raises(ValueError, match="shared tensor differs.*model_sigma.0.weight"):
        converter.build_bundle(sources, tmp_path / "bundle.safetensors", contract)

def test_independent_conversions_are_byte_identical(tmp_path):
    sources, contract = write_legacy_sources(tmp_path)
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    converter.build_bundle(sources, first, contract)
    converter.build_bundle(sources, second, contract)
    assert first.read_bytes() == second.read_bytes()
```

- [ ] **Step 3: Run converter tests and verify RED**

```bash
conda run -n tsac_env python -m pytest tests/test_gamma_ensemble_converter.py -v
```

Expected: import failure because the converter and shared canonicalizer do not
exist.

- [ ] **Step 4: Extract the proven canonical-header helper**

Move `_canonical_json_value` and `_canonicalize_safetensors_header` without
behavior changes from `scripts/build_smi_ted_inference_asset.py` into
`scripts/_safetensors_canonical.py`, then import them back into the SMI-TED
converter. Keep all constants needed by the helper in the focused module.

Both converters must remain usable when imported by tests and when invoked by
their existing direct script paths. Use package-aware imports rather than
catching an arbitrary import failure:

```python
if __package__:
    from scripts._safetensors_canonical import _canonicalize_safetensors_header
else:
    from _safetensors_canonical import _canonicalize_safetensors_header
```

Update test imports only where necessary; do not regenerate the SMI-TED
candidate or alter its manifest digest.

- [ ] **Step 5: Verify the SMI-TED refactor before adding new behavior**

```bash
conda run -n tsac_env python -m pytest tests/test_smi_ted_converter.py -v
```

Expected: all existing SMI-TED converter tests pass.

- [ ] **Step 6: Implement contract parsing and source verification**

Use exact JSON key checks, `Path.is_file()`, `Path.is_symlink()`, streaming
SHA-256 before `torch.load`, and restricted CPU loading:

```python
state = torch.load(str(source_path), map_location="cpu", weights_only=True)
```

Require exactly the ten approved numbered `.ckpt` names in the selected source
directory. Derive the 33 shared and six head keys from the exact
`Prf_to_Seg_Model().state_dict()` key set and the approved `model_final.`
prefix, then compare key, shape, dtype, and value constraints before mapping.

- [ ] **Step 7: Implement atomic deterministic save and strict self-check**

Build an insertion-ordered mapping with lexical keys, save to a same-directory
temporary file, canonicalize its header, load it with `safetensors.torch`, and
compare all 93 values exactly. Require total tensor bytes 5,202,560 for the
production ten-member architecture, but allow synthetic test architectures to
verify their own calculated total.

```python
metadata = {
    "asset_name": "gamma-tuned-ensemble",
    "format_version": "1",
    "member_count": "10",
    "shared_tensor_count": "33",
    "head_tensor_count": "60",
    "source_manifest_bundle_version": "1.0.0",
}
save_file(bundle_state, temporary_path, metadata=metadata)
_canonicalize_safetensors_header(temporary_path)
```

Atomically install with `os.replace` only after strict verification. On any
failure, unlink only the converter-owned temporary file and preserve sources
and pre-existing targets.

- [ ] **Step 8: Run focused converter and SMI-TED regression tests**

```bash
conda run -n tsac_env python -m pytest \
  tests/test_gamma_ensemble_converter.py tests/test_smi_ted_converter.py -v
```

Expected: all pass, including independent byte identity and unchanged
SMI-TED canonical-header behavior.

- [ ] **Step 9: Commit Task 2**

```bash
git add scripts/_safetensors_canonical.py \
  scripts/gamma_ensemble_sources.json \
  scripts/build_gamma_ensemble_asset.py \
  scripts/build_smi_ted_inference_asset.py \
  tests/test_gamma_ensemble_converter.py tests/test_smi_ted_converter.py
git commit -m "feat: convert gamma ensemble deterministically"
```

---

### Task 3: Add strict loading and generate the real parity candidate

**Files:**
- Create: `src/tennetsac/gamma_ensemble.py`
- Create: `tests/test_gamma_ensemble_loader.py`
- Create: `tests/integration/test_gamma_ensemble_asset.py`
- Create: `tests/fixtures/gamma_ensemble_golden.json`
- Create: `scripts/generate_gamma_ensemble_golden.py`
- Modify: `scripts/build_gamma_ensemble_asset.py`

**Interfaces:**
- Consumes: the Task 1 architecture and Task 2 93-tensor bundle.
- Produces: `GammaEnsembleLoadError(ValueError)` and `load_gamma_ensemble(path: Path, expected_sha256: str) -> GammaEnsemble`.
- Produces a real candidate at `/private/tmp/tennetsac-gamma-ensemble-v1-candidate-20260908-r1/gamma-ensemble-v1.safetensors` and a hash-verified source backup at `/private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908`.

- [ ] **Step 1: Add RED strict-loader tests**

Cover valid loading, hash-before-parse ordering, non-regular/symlink paths,
uppercase or malformed expected digests, missing/extra/duplicate keys, wrong
metadata, wrong tensor shape/dtype/count/bytes, and preservation of underlying
causes.

```python
@pytest.fixture
def valid_bundle(tmp_path):
    path = tmp_path / BUNDLE_FILENAME
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in GammaEnsemble().eval().state_dict().items()
    }
    save_file(state, path, metadata=BUNDLE_METADATA)
    _canonicalize_safetensors_header(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest

def test_loader_rejects_digest_mismatch_before_safetensors_open(
    monkeypatch, valid_bundle
):
    bundle, _ = valid_bundle
    opens = []
    monkeypatch.setattr(loader, "safe_open", lambda *args, **kwargs: opens.append(args))
    with pytest.raises(GammaEnsembleLoadError, match="sha256 mismatch"):
        load_gamma_ensemble(bundle, "0" * 64)
    assert opens == []

def test_loader_returns_exact_eval_ensemble(valid_bundle):
    bundle, digest = valid_bundle
    model = load_gamma_ensemble(bundle, digest)
    assert isinstance(model, GammaEnsemble)
    assert model.training is False
    assert len(model.heads) == 10
    assert len(model.state_dict()) == 93
```

- [ ] **Step 2: Run strict-loader tests and verify RED**

```bash
conda run -n tsac_env python -m pytest tests/test_gamma_ensemble_loader.py -v
```

Expected: import failure because `tennetsac.gamma_ensemble` does not exist.

- [ ] **Step 3: Implement hash, header, metadata, and strict state validation**

Define the exact constants once in `src/tennetsac/gamma_ensemble.py`:

```python
BUNDLE_FILENAME = "gamma-ensemble-v1.safetensors"
BUNDLE_TENSOR_COUNT = 93
BUNDLE_TENSOR_BYTES = 5_202_560
BUNDLE_METADATA = {
    "asset_name": "gamma-tuned-ensemble",
    "format_version": "1",
    "member_count": "10",
    "shared_tensor_count": "33",
    "head_tensor_count": "60",
    "source_manifest_bundle_version": "1.0.0",
}
```

Stream-hash the file before parsing. Read the raw safetensors header with a
JSON `object_pairs_hook` that rejects duplicate names, compare exact metadata
and tensor names, then load tensors on CPU. Compare every loaded shape and
dtype with a fresh `GammaEnsemble().state_dict()` and call
`load_state_dict(..., strict=True)`.

- [ ] **Step 4: Run focused loader and architecture tests**

```bash
conda run -n tsac_env python -m pytest \
  tests/test_gamma_ensemble_loader.py tests/test_gamma_ensemble.py -v
```

Expected: all pass.

- [ ] **Step 5: Create and verify a source backup before any package deletion**

Require the destination to be absent; copy only the ten source files, retain
mode `0644`, and verify every copied hash against
`scripts/gamma_ensemble_sources.json`.

```bash
test ! -e /private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908
mkdir -p /private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908
cp src/tennetsac/ckpt_files/fine-tuned/{1,2,3,4,5,6,7,8,9,10}.ckpt \
  /private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908/
```

Use a read-only verification command to compare all ten copied digests with
the contract before proceeding.

- [ ] **Step 6: Generate two independent real candidates**

Require both output roots to be absent and run:

```bash
conda run -n tsac_env python scripts/build_gamma_ensemble_asset.py \
  src/tennetsac/ckpt_files/fine-tuned \
  /private/tmp/tennetsac-gamma-ensemble-v1-candidate-20260908-r1/gamma-ensemble-v1.safetensors \
  --source-contract scripts/gamma_ensemble_sources.json

conda run -n tsac_env python scripts/build_gamma_ensemble_asset.py \
  /private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908 \
  /private/tmp/tennetsac-gamma-ensemble-v1-candidate-20260908-r2/gamma-ensemble-v1.safetensors \
  --source-contract scripts/gamma_ensemble_sources.json
```

Compare the complete files with `cmp`, compute both SHA-256 digests, inspect
exact metadata/count/bytes with the strict loader, and record the one observed
digest. Do not type or predict the digest before this gate.

- [ ] **Step 7: Create the real pre-migration golden fixture**

Use fixed positive sigma profiles, `298.15 K` and `343.15 K`, and explicit
legacy checkpoint members to record:

- all ten segment-activity gradients for at least two sigma inputs;
- tuned means;
- population standard deviations;
- one result for each numbered member; and
- fixed binary and three-component final `ln gamma` mean/std results using
  deterministic injected sigma profiles, areas, and volumes.

Write decimal values with enough precision to enforce `rtol=1e-5` and
`atol=1e-6`. The fixture records generator command, source commit, every source
digest, PyTorch version, and `ddof=0`. Implement the generator as a
no-overwrite CLI and run exactly:

```bash
test ! -e tests/fixtures/gamma_ensemble_golden.json
conda run -n tsac_env python scripts/generate_gamma_ensemble_golden.py \
  --legacy-dir /private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908 \
  --source-contract scripts/gamma_ensemble_sources.json \
  --output tests/fixtures/gamma_ensemble_golden.json
```

Run the generator a second time into a new `/private/tmp` path and compare the
JSON bytes to prove deterministic ordering and float serialization.

- [ ] **Step 8: Add and run the real parity integration test**

The integration test reads `TENNETSAC_LEGACY_GAMMA_DIR` and
`TENNETSAC_GAMMA_ENSEMBLE`; it skips before loading PyTorch if either is
absent. When present, it hashes all sources, strictly loads the candidate, and
compares all ten member gradients, ensemble mean, and population std across
the fixed fixture cases.

```bash
TENNETSAC_LEGACY_GAMMA_DIR=/private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908 \
TENNETSAC_GAMMA_ENSEMBLE=/private/tmp/tennetsac-gamma-ensemble-v1-candidate-20260908-r1/gamma-ensemble-v1.safetensors \
conda run -n tsac_env python -m pytest \
  tests/integration/test_gamma_ensemble_asset.py -v
```

Expected: real parity passes for every member, mean, std, tensor count, tensor
bytes, metadata, and observed digest.

- [ ] **Step 9: Commit Task 3 without committing the temporary candidate**

```bash
git add src/tennetsac/gamma_ensemble.py \
  scripts/build_gamma_ensemble_asset.py scripts/generate_gamma_ensemble_golden.py \
  tests/test_gamma_ensemble_loader.py \
  tests/integration/test_gamma_ensemble_asset.py \
  tests/fixtures/gamma_ensemble_golden.json
git commit -m "feat: validate gamma ensemble bundles"
```

---

### Task 4: Integrate the bundle, manifest, package inventory, and runtime

**Files:**
- Create: `src/tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors` from the verified Task 3 candidate
- Delete: `src/tennetsac/ckpt_files/fine-tuned/1.ckpt` through `10.ckpt`
- Modify: `src/tennetsac/model_manifest.json`
- Modify: `src/tennetsac/_manifest_schema.py`
- Modify: `src/tennetsac/model_manifest.py`
- Modify: `src/tennetsac/utils/model_io.py`
- Modify: `src/tennetsac/runtime.py`
- Modify: `src/tennetsac/core.py`
- Modify: `src/tennetsac/utils/property.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_model_manifest.py`
- Create: `tests/test_model_io.py`
- Modify: `tests/test_runtime.py`
- Modify: `tests/test_gamma_ensemble.py`

**Interfaces:**
- Consumes: `load_gamma_ensemble(path, expected_sha256)` and the observed Task 3 digest.
- Produces: `Runtime.gamma_ensemble: GammaEnsemble` and temporary mean-only `ensemble_predictor(sigma, temperature, *, member_index=None, return_members=False)` integration used by Task 5.
- Removes: runtime use of `load_all_Gamma_models` and the ten-model tuple.

- [ ] **Step 1: Add RED manifest and runtime inventory tests**

Change literal manifest expectations to four bundled artifacts and bundle
version `2.0.0`. Only after Task 3 prints the real lowercase SHA-256, paste
that complete 64-character value literally into both the test and manifest;
do not introduce a symbolic digest constant or calculate the expected value
from the file under test. Obtain and validate the value with:

```bash
candidate_digest=$(shasum -a 256 \
  /private/tmp/tennetsac-gamma-ensemble-v1-candidate-20260908-r1/gamma-ensemble-v1.safetensors \
  | awk '{print $1}')
test "${#candidate_digest}" -eq 64
printf '%s\n' "$candidate_digest"
```

The four literal entries are the unchanged `gamma-base`, `geometry`, and
`sigma-profile` records followed by `gamma-tuned-ensemble` at
`ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors`, with distribution
`bundled` and the printed digest. Review the staged Python and JSON to confirm
the printed 64 characters appear literally in both files.

Update runtime tests to require one `load_gamma_ensemble` call with the
materialized bundle path and manifest digest, a `Runtime.gamma_ensemble`
field, and no `load_all_Gamma_models` call. Add missing-resource tests naming
`fine-tuned/gamma-ensemble-v1.safetensors`.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
conda run -n tsac_env python -m pytest \
  tests/test_model_manifest.py tests/test_runtime.py tests/test_model_io.py -v
```

Expected: failures show the old 13-artifact manifest, ten-file runtime
inventory, and absent ensemble loader wiring.

- [ ] **Step 3: Install the verified generated artifact without rebuilding**

First re-hash the r1 candidate and compare it with r2 and the digest used by
the RED tests. Require the package target to be absent, then copy the already
verified r1 bytes to the exact target and set mode `0644`. Run the strict
loader against the copied target before removing any source.

- [ ] **Step 4: Update manifest schema and exact inventory**

Replace the hard-coded thirteen-checkpoint path set with the exact four-path
set. Require `bundle_version == "2.0.0"` and require the ensemble entry's exact
name/path/distribution and observed digest shape. Keep external-model and
tokenizer records byte-for-value unchanged.

Add `bundled_artifact(name: str) -> dict` to `model_manifest.py` so runtime
obtains the bundle path and digest by logical name rather than duplicating
manifest literals.

- [ ] **Step 5: Wire strict runtime loading**

Change `_CHECKPOINT_RESOURCES` to:

```python
_CHECKPOINT_RESOURCES = (
    "base.ckpt",
    "geo.ckpt",
    "prf.ckpt",
    "fine-tuned/gamma-ensemble-v1.safetensors",
)
```

Materialize the resource through the existing `_checkpoint_path` context,
look up `bundled_artifact("gamma-tuned-ensemble")`, and call:

```python
gamma_ensemble = load_gamma_ensemble(
    checkpoint_root / ensemble_entry["path"].removeprefix("ckpt_files/"),
    expected_sha256=ensemble_entry["sha256"],
)
```

Wrap failures as `RuntimeError("Failed to initialize gamma ensemble")` with
the original cause. Replace `gamma_finetuned_models` in the frozen `Runtime`
dataclass with `gamma_ensemble`.

- [ ] **Step 6: Preserve mean-only internal behavior during migration**

Update `ensemble_predictor` to call `runtime.gamma_ensemble.predict_segac` and
return a CPU tensor. Update `ensemble_segac` to accept the ensemble object
instead of a list and delegate to its mean path. Do not add the public
`return_std` default until Task 5.

- [ ] **Step 7: Change package data and remove redundant sources**

Replace `ckpt_files/fine-tuned/*.ckpt` in package data with the literal
`ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors`. Reverify the backup and
installed bundle hashes, then remove the ten package copies with `git rm`.
Never remove the verified `/private/tmp` backup.

- [ ] **Step 8: Run manifest, runtime, loader, and existing API tests**

```bash
conda run -n tsac_env python -m pytest \
  tests/test_model_manifest.py tests/test_model_io.py tests/test_runtime.py \
  tests/test_gamma_ensemble_loader.py tests/test_gamma_ensemble.py \
  tests/test_api_contract.py -v
```

Expected: all pass with the historical mean-only public API still intact at
this intermediate commit.

- [ ] **Step 9: Re-run real parity using the preserved backup**

```bash
TENNETSAC_LEGACY_GAMMA_DIR=/private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908 \
TENNETSAC_GAMMA_ENSEMBLE=src/tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors \
conda run -n tsac_env python -m pytest \
  tests/integration/test_gamma_ensemble_asset.py -v
```

Expected: all real member and ensemble comparisons pass after source deletion.

- [ ] **Step 10: Commit Task 4**

Stage the exact code, manifest, generated bundle, and ten deletions; inspect
`git diff --cached --stat` and `git diff --cached --check`, then commit:

```bash
git commit -m "feat: package shared gamma ensemble"
```

---

### Task 5: Add default mean/std results and numbered member selection

**Files:**
- Modify: `src/tennetsac/core.py`
- Modify: `src/tennetsac/utils/property.py`
- Modify: `tests/test_api_contract.py`
- Create: `tests/test_gamma_statistics.py`
- Modify: `tests/integration/test_full_model.py`

**Interfaces:**
- Consumes: `GammaEnsemble.predict_segac(..., member_index=None, return_members=False)` from Task 1 and `Runtime.gamma_ensemble` from Task 4.
- Produces: public `binary_lng(..., version="tuned", return_std=True)` and `multi_lng(..., version="tuned", return_std=True)`.
- Produces: accepted versions `"base"`, `"tuned"`, and `"1"` through `"10"`.
- Produces: tuned binary four-list stats return and tuned multi two-list stats return; `return_std=False` preserves historical shapes.

- [ ] **Step 1: Add RED public signature, return-shape, and dispatch tests**

Update exact signature expectations and cover all accepted versions:

```python
def test_binary_lng_returns_statistics_by_default(monkeypatch):
    monkeypatch.setattr(
        core,
        "calc_ln_gamma_binary",
        lambda *args, **kwargs: tuple(
            np.array(values) for values in ([1, 2], [3, 4], [.1, .2], [.3, .4])
        ),
    )
    assert tennetsac.binary_lng(["CCO", "O"], 298.15, [.25, .75]) == (
        [1, 2], [3, 4], [.1, .2], [.3, .4]
    )

@pytest.mark.parametrize("version", ["base", *map(str, range(1, 11))])
def test_single_model_versions_require_mean_only(version):
    with pytest.raises(ValueError, match="return_std=False"):
        tennetsac.binary_lng(["CCO", "O"], 298.15, [.5], version=version)
```

Assert `version="1"` maps to zero-based `member_index=0`, `"10"` maps to 9,
integers are rejected, unknown strings list the accepted forms, and validation
happens before sigma-profile model work.

- [ ] **Step 2: Add RED final-statistics reference tests**

Use a fake predictor returning a fixed `(members, batch, 51)` tensor. Compute
each member's complete binary and multi `ln gamma` explicitly, then require
the implementation to return `numpy.mean(axis=0)` and
`numpy.std(axis=0, ddof=0)`. Include a case where pure and mixture member
ordering matters, proving that std is not derived from separately aggregated
segac standard deviations.

- [ ] **Step 3: Run API/statistics tests and verify RED**

```bash
conda run -n tsac_env python -m pytest \
  tests/test_api_contract.py tests/test_gamma_statistics.py -v
```

Expected: failures identify absent `return_std`, unsupported numbered
versions, and the old mean-only calculations.

- [ ] **Step 4: Implement version dispatch and predictor modes**

Use a single validator in `core.py`:

```python
def select_gamma_predictor(version: str, *, return_std: bool):
    accepted = {"base", "tuned", *(str(i) for i in range(1, 11))}
    if type(version) is not str or version not in accepted:
        raise ValueError("version must be 'base', 'tuned', or a string from '1' to '10'")
    if type(return_std) is not bool:
        raise TypeError("return_std must be a bool")
    if version == "tuned":
        return functools.partial(ensemble_predictor, return_members=return_std)
    if return_std:
        raise ValueError(
            "return_std=True requires version='tuned'; pass return_std=False "
            "for version='base' or a numbered fine-tuned member"
        )
    if version == "base":
        return single_model_predictor
    return functools.partial(ensemble_predictor, member_index=int(version) - 1)
```

Validate the version before the standard-deviation combination so an unknown
version always produces the version error even though `return_std` defaults to
true. `ensemble_predictor` delegates to the one runtime ensemble and returns
CPU tensors without converting member arrays to NumPy prematurely.

- [ ] **Step 5: Implement member-preserving final statistics**

Add `return_std=False` to the internal `calc_ln_gamma` and
`calc_ln_gamma_binary` functions. In statistics mode, retain a leading member
dimension through pure and mixture segac calculations, compute each member's
complete residual and combinatorial result, then aggregate only at the end:

```python
member_values = torch.stack(per_component_or_grid_values, dim=-1)
mean = member_values.mean(dim=0)
std = member_values.std(dim=0, unbiased=False)
return mean.cpu().numpy(), std.cpu().numpy()
```

For binary results, return mean component 1, mean component 2, std component
1, std component 2 in that exact order. Mean-only mode follows the existing
equations and returns the historical arrays.

- [ ] **Step 6: Update public functions and preserve NRTL behavior**

Add `return_std: bool = True` after `version` in both public signatures and
document the intentional `v0.2.0` return change. Convert every returned NumPy
array to a Python list. Change the internal `fit_nrtl` call to:

```python
l1, l2 = binary_lng(
    smiles_pair,
    T,
    x1_eval.tolist(),
    return_std=False,
)
```

Do not add uncertainty values to `fit_nrtl` or change its public signature.

- [ ] **Step 7: Run focused public and numerical tests**

```bash
conda run -n tsac_env python -m pytest \
  tests/test_api_contract.py tests/test_gamma_statistics.py \
  tests/test_gamma_ensemble.py -v
```

Expected: default stats, mean-only compatibility, numbered heads, population
std, and one-trunk-forward assertions all pass.

- [ ] **Step 8: Update and run full-model integration expectations**

Change old calls that unpack two binary arrays or one multi array to pass
`return_std=False`. Add cached-model integration assertions for the new
default: finite non-negative standard deviations of the correct shapes, plus
mean equality against the existing PyPI golden tolerance.

```bash
TENNETSAC_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run -n tsac_env python -m pytest tests/integration/test_full_model.py -v
```

Expected: pass when pinned ChemBERTa2 and the SMI-TED cache are available;
otherwise only the existing explicit cache skips are allowed.

- [ ] **Step 9: Commit Task 5**

```bash
git add src/tennetsac/core.py src/tennetsac/utils/property.py \
  tests/test_api_contract.py tests/test_gamma_statistics.py \
  tests/integration/test_full_model.py
git commit -m "feat: return gamma ensemble uncertainty"
```

---

### Task 6: Enforce the new distribution and document the breaking API

**Files:**
- Modify: `scripts/verify_distribution.py`
- Modify: `tests/test_distribution.py`
- Create: `docs/model-assets/gamma-ensemble-v1.md`
- Modify: `README.md`
- Modify: `src/tennetsac/assets/README.md`
- Modify: `tests/test_documentation.py`

**Interfaces:**
- Consumes: the exact manifest and package asset from Task 4 and public API from Task 5.
- Produces: archives containing exactly three `.ckpt` files and one approved internal safetensors file, while rejecting every other `.pt` or `.safetensors` file.
- Produces user documentation for default stats, mean-only compatibility, numbered members, size reduction, and provenance.

- [ ] **Step 1: Add RED archive allowlist tests**

Change fixture inventory to:

```python
MODEL_ASSETS = {
    "base.ckpt": "gamma-base",
    "geo.ckpt": "geometry",
    "prf.ckpt": "sigma-profile",
    "fine-tuned/gamma-ensemble-v1.safetensors": "gamma-tuned-ensemble",
}
```

Require valid wheel/sdist fixtures with the declared safetensors member to
pass. Keep explicit rejection tests for SMI-TED safetensors and add rejection
for an undeclared `fine-tuned/other.safetensors`, any `.pt`, any old numbered
`.ckpt`, a missing ensemble bundle, and a digest mismatch.

- [ ] **Step 2: Run distribution tests and verify RED**

```bash
conda run -n tsac_env python -m pytest tests/test_distribution.py -v
```

Expected: the current global safetensors ban rejects the newly approved
fixture and the old checkpoint constants disagree.

- [ ] **Step 3: Implement exact internal-safetensors allowlisting**

Allow only normalized archive paths ending in the exact declared internal
path `ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors`. Continue rejecting
every other `.safetensors` and `.pt` before archive content is trusted.
Extend undeclared-weight detection to both `.ckpt` and `.safetensors`, and
require every packaged model-weight member to appear exactly once in the
validated manifest with a matching digest.

- [ ] **Step 4: Update documentation tests and verify RED**

Require README examples containing:

```python
mean_1, mean_2, std_1, std_2 = binary_lng(...)
mean = multi_lng(..., return_std=False)
member_7 = multi_lng(..., version="7", return_std=False)
```

Also require the exact bundle filename, 10-member population std definition,
approximately 5.2 MB versus 37.3 MB size statement, `version="base"`/numbered
std restriction, and a `v0.2.0` breaking-change notice.

- [ ] **Step 5: Write user and provenance documentation**

Update README API examples and project structure. Change the reserved-assets
README only to clarify that the gamma ensemble is a bundled project-owned
asset, not an external model. Create
`docs/model-assets/gamma-ensemble-v1.md` to record that the bundle is a
lossless structural repack of the project's ten existing fine-tuned
checkpoints, including source commit and hashes, converter command/commit,
tensor mapping/count/bytes, final file hash/bytes, parity command, and
benchmark summary. Do not change `THIRD_PARTY_NOTICES.md`: this project-owned
asset is not a third-party licensing notice.

- [ ] **Step 6: Run distribution and documentation tests**

```bash
conda run -n tsac_env python -m pytest \
  tests/test_distribution.py tests/test_documentation.py \
  tests/test_package_metadata.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit Task 6**

```bash
git add scripts/verify_distribution.py tests/test_distribution.py \
  docs/model-assets/gamma-ensemble-v1.md \
  README.md src/tennetsac/assets/README.md \
  tests/test_documentation.py
git commit -m "docs: document gamma ensemble statistics"
```

---

### Task 7: Benchmark, build, run full regression, and stop at review

**Files:**
- Create: `scripts/benchmark_gamma_ensemble.py`
- Create: `docs/benchmarks/2026-09-08-gamma-ensemble.md`
- Modify: tests only if the final gate reveals a demonstrated regression in the approved behavior

**Interfaces:**
- Consumes: preserved legacy source directory and the integrated strict bundle.
- Produces: reproducible JSON/text benchmark output with legacy mean, new mean, new mean+std, tensor bytes, file bytes, and speed ratios.
- Produces: a locally reviewed branch; no external release operations.

- [ ] **Step 1: Add the controlled benchmark script**

The CLI accepts `--legacy-dir`, `--bundle`, `--iterations`, `--warmup`,
`--threads`, and `--output`. It verifies source/bundle hashes before timing,
uses one fixed `(1, 51)` positive sigma and `298.15 K`, performs garbage
collection between cases, uses `time.perf_counter_ns`, and reports medians
from repeated batches rather than a single call.

```bash
conda run -n tsac_env python scripts/benchmark_gamma_ensemble.py \
  --legacy-dir /private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908 \
  --bundle src/tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors \
  --threads 1 --warmup 20 --iterations 500 \
  --output /private/tmp/tennetsac-gamma-ensemble-benchmark-20260908.json
```

The script exits nonzero if numerical parity fails or the new mean path is
less than 2.0 times faster than the legacy mean on this controlled host. It
reports but does not impose an arbitrary speed threshold on mean+std.

- [ ] **Step 2: Record benchmark evidence**

Run the benchmark at least three times, record the median run in
`docs/benchmarks/2026-09-08-gamma-ensemble.md`, and include:

- Python, PyTorch, platform, processor, and thread count;
- legacy mean median and dispersion;
- new mean median and speed ratio;
- new default mean+std median and speed ratio relative to an explicit legacy
  ten-member mean+std reference;
- legacy/new tensor bytes and percentage reduction;
- ten-source/new-bundle file bytes and percentage reduction; and
- exact bundle SHA-256.

- [ ] **Step 3: Run the complete local non-integration suite**

```bash
conda run -n tsac_env python -m pytest -m "not integration" -v
```

Expected: all tests pass; only previously documented dependency deprecations
are acceptable.

- [ ] **Step 4: Run exact real-asset and full-model integration gates**

```bash
TENNETSAC_LEGACY_GAMMA_DIR=/private/tmp/tennetsac-gamma-ensemble-v1-sources-20260908 \
TENNETSAC_GAMMA_ENSEMBLE=src/tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors \
TENNETSAC_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run -n tsac_env python -m pytest \
  tests/integration/test_gamma_ensemble_asset.py \
  tests/integration/test_smi_ted_asset.py \
  tests/integration/test_full_model.py -v
```

Expected: gamma real parity and cached full-model tests pass. Public-download
tests remain gated unless their existing explicit flag is supplied; this task
must not redownload or republish SMI-TED.

- [ ] **Step 5: Build and verify wheel and sdist**

Use a new absent temporary output directory:

```bash
conda run -n tsac_env python -m build --no-isolation \
  --outdir /private/tmp/tennetsac-gamma-ensemble-dist-20260908
conda run -n tsac_env python -m twine check \
  /private/tmp/tennetsac-gamma-ensemble-dist-20260908/*
conda run -n tsac_env python scripts/verify_distribution.py \
  /private/tmp/tennetsac-gamma-ensemble-dist-20260908/*
```

Independently inspect both archives and require exactly three `.ckpt` members,
one `gamma-ensemble-v1.safetensors`, zero numbered fine-tuned checkpoints,
zero `.pt`, exact manifest digests, and at least a 30 MB reduction in packaged
fine-tuned assets.

- [ ] **Step 6: Commit benchmark tooling and report**

```bash
git add scripts/benchmark_gamma_ensemble.py \
  docs/benchmarks/2026-09-08-gamma-ensemble.md
git commit -m "perf: benchmark shared gamma ensemble"
```

- [ ] **Step 7: Perform final branch audit**

Run:

```bash
git diff --check 7711c6bd3dff7e2d7cff00590062f9e1a0ca605a...HEAD
git status --short --branch
git log --oneline --decorate 7711c6bd3dff7e2d7cff00590062f9e1a0ca605a..HEAD
git diff --stat 7711c6bd3dff7e2d7cff00590062f9e1a0ca605a...HEAD
git ls-files 'src/tennetsac/ckpt_files/fine-tuned/*'
```

Require a clean branch, only the approved fine-tuned bundle path, no tracked
temporary candidates, no `.superpowers` files, and no unrelated changes.

- [ ] **Step 8: Request independent code review and stop**

Review the complete range from
`7711c6bd3dff7e2d7cff00590062f9e1a0ca605a` through `HEAD` for mathematical
correctness, autograd graph retention, member ordering, default API break,
strict artifact validation, deterministic conversion, package contents, and
benchmark methodology. Address findings with focused tests and commits, rerun
the affected gates, then rerun the complete final verification.

Report the final commit, exact bundle digest/size, test counts, old/new archive
sizes, all three benchmark medians, and open limitations. Stop before merge,
push, tag, GitHub Release, or PyPI publication and ask for the next explicit
integration decision.
