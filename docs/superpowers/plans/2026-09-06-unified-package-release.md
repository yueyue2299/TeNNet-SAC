# Unified Package and Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `TeNNet-SAC` into the single installable and releasable source for the `tennetsac` PyPI package, with tag-derived versions, lazy model loading, reproducible model provenance, and manually protected PyPI publication.

**Architecture:** Move the package into `src/tennetsac`, preserve the currently published API implementation as the behavioral baseline, and overlay the already-fixed SMI-TED tokenizer. A cached runtime object performs model initialization only on the first prediction call; Git tags drive Python versions, while a checked model manifest records checkpoint hashes and pinned Hugging Face revisions.

**Tech Stack:** Python 3.10-3.12, setuptools, setuptools-scm, pytest, PyTorch, Transformers 4.36.2, Hugging Face Hub, GitHub Actions, PyPI Trusted Publishing.

**Spec:** `docs/superpowers/specs/2026-09-06-unified-package-release-design.md`

## Global Constraints

- `TeNNet-SAC` is the only maintained source repository; do not merge the `TSAC_pypi` Git history.
- Preserve `import tennetsac` and the public callables `profile`, `binary_lng`, `multi_lng`, `fit_nrtl`, and `plot_nrtl_fitting`.
- Preserve the published function arguments and return types in this migration.
- Use the old PyPI `core.py` and `utils/property.py` behavior as the baseline; do not include the GitHub mean/standard-deviation ensemble API yet.
- Overlay the SMI-TED tokenizer and loader from GitHub commit `df401e4`.
- Keep `transformers==4.36.2` as the package runtime dependency; Transformers v5 is tested only for the standalone SMI-TED tokenizer.
- Importing `tennetsac` must not access the network or initialize model weights.
- Keep all existing checkpoint bytes unchanged.
- Do not bundle the approximately 1.15 GB SMI-TED checkpoint in the wheel.
- Reserve `src/tennetsac/assets/chemberta2/`, but do not bundle ChemBERTa2 files in this work.
- A tag build creates a draft GitHub Release and never publishes to PyPI.
- Manual PyPI publication must reuse the exact artifacts from the draft GitHub Release and must not rebuild them.

---

## File Structure

### Package and runtime

- `pyproject.toml`: build backend, tag-derived version, dependencies, package discovery, package data, pytest configuration.
- `MANIFEST.in`: limits source distributions to package/build inputs and excludes repository-only material.
- `src/tennetsac/__init__.py`: version export and lazy public API resolution only.
- `src/tennetsac/core.py`: existing public API behavior, wired to `get_runtime()` instead of module-level model instances.
- `src/tennetsac/runtime.py`: cached heavyweight model construction and package-resource resolution.
- `src/tennetsac/model_manifest.py`: load, query, and verify the JSON model manifest.
- `src/tennetsac/model_manifest.json`: bundle version, exact checkpoint hashes, external model revisions, tokenizer format.
- `src/tennetsac/assets/README.md`: documents the future offline ChemBERTa2 directory.
- `src/tennetsac/models/`: checkpoint-compatible PyTorch architectures.
- `src/tennetsac/utils/`: published property calculations, embedding adapters, model I/O, plotting, and SMILES helpers.
- `src/tennetsac/smi_ted_light/`: fixed SMI-TED tokenizer, loader, vocabulary, and transformer implementation.
- `src/tennetsac/ckpt_files/`: unchanged TeNNet-SAC checkpoint bytes.

### Tests and verification

- `tests/test_package_metadata.py`: version exposure, public API names, and import isolation.
- `tests/test_runtime.py`: one-time runtime construction, local resource paths, and actionable failures.
- `tests/test_api_contract.py`: published signatures, validation behavior, and return conversions without loading real models.
- `tests/test_model_manifest.py`: schema, revisions, artifact set, and SHA256 verification.
- `tests/test_embeddings.py`: manifest-driven revision forwarding for ChemBERTa2 and SMI-TED.
- `tests/test_smi_ted_loading.py`: package-relative SMI-TED loading and pinned fallback requests.
- `tests/test_smi_ted_tokenizer.py`: golden token IDs, independent of Transformers internals.
- `tests/integration/test_full_model.py`: opt-in full-model equivalence and smoke checks using local caches only.
- `scripts/verify_distribution.py`: wheel/sdist contents and metadata validation.
- `scripts/verify_release_version.py`: exact tag, artifact metadata, and installed version equality.

### Automation and documentation

- `.github/workflows/ci.yml`: supported-Python unit tests, tokenizer compatibility, build checks, and installed-wheel smoke test.
- `.github/workflows/release-build.yml`: tag validation, one-time artifact build, release checks, and draft GitHub Release.
- `.github/workflows/publish-pypi.yml`: protected manual publication of existing release assets.
- `examples/TeNNetSAC.ipynb`: example notebook using the installed package.
- `README.md`: unified install, source-development, model-loading, and release-source guidance.
- `.gitignore`: ignores local build/test artifacts while retaining workflow YAML files.

---

### Task 1: Add tag-derived package metadata

**Files:**
- Create: `pyproject.toml`
- Create: `src/tennetsac/__init__.py`
- Test: `tests/test_package_metadata.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Git tags in the form `v<PEP-440-version>`.
- Produces: `tennetsac.__version__: str`, `tennetsac.__all__: list[str]`, and a setuptools-scm generated `src/tennetsac/_version.py`.

- [ ] **Step 1: Write failing metadata tests**

Add tests that import the package from an installed/editable environment, assert a non-empty PEP 440 version, and confirm import has not loaded runtime modules:

```python
from packaging.version import Version
import sys


def test_package_exposes_pep440_version():
    import tennetsac

    assert Version(tennetsac.__version__)


def test_plain_import_does_not_load_model_runtime():
    import tennetsac

    assert "tennetsac.runtime" not in sys.modules
    assert "tennetsac.core" not in sys.modules
    assert set(tennetsac.__all__) == {
        "profile",
        "binary_lng",
        "multi_lng",
        "fit_nrtl",
        "plot_nrtl_fitting",
        "__version__",
    }
```

- [ ] **Step 2: Run the tests and confirm the package is absent**

Run: `python -m pytest tests/test_package_metadata.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'tennetsac'`.

- [ ] **Step 3: Add the build configuration**

Create `pyproject.toml` with these exact version and package-discovery settings:

```toml
[build-system]
requires = ["setuptools>=77", "setuptools-scm>=8"]
build-backend = "setuptools.build_meta"

[project]
name = "tennetsac"
dynamic = ["version"]
description = "Thermodynamics-embedded Neural Network for Segment Activity Coefficients"
readme = "README.md"
requires-python = ">=3.10,<3.13"
license = "MIT"
authors = [{ name = "Yue Yang" }]
dependencies = [
  "numpy>=1.25,<2",
  "pandas<2.2",
  "pyarrow<15",
  "scikit-learn",
  "scipy",
  "rdkit",
  "matplotlib",
  "tqdm",
  "transformers==4.36.2",
  "huggingface-hub",
  "safetensors",
  "tokenizers",
  "accelerate",
  "torch>=2.1.2",
]

[project.optional-dependencies]
test = ["pytest>=8", "packaging", "PyYAML>=6"]
build = ["build>=1.2", "twine>=5"]
dev = [
  "pytest>=8",
  "packaging",
  "PyYAML>=6",
  "build>=1.2",
  "twine>=5",
  "jupyterlab",
]

[tool.setuptools]
package-dir = { "" = "src" }
include-package-data = true

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
tennetsac = [
  "model_manifest.json",
  "assets/*.md",
  "smi_ted_light/*.txt",
  "ckpt_files/*.ckpt",
  "ckpt_files/fine-tuned/*.ckpt",
]

[tool.setuptools_scm]
write_to = "src/tennetsac/_version.py"
version_scheme = "no-guess-dev"
local_scheme = "node-and-date"
fallback_version = "0.0.dev0"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: requires locally available full model weights"]
```

Do not restore the old `torchvision` or `torchaudio` dependencies: no package source imports them. Add `matplotlib`, which the published `fit_nrtl` plotting API imports but the old PyPI metadata omitted.

- [ ] **Step 4: Add a lightweight package initializer**

Create `src/tennetsac/__init__.py` with only version loading and lazy API lookup:

```python
from importlib import import_module

from ._version import __version__

__all__ = [
    "profile",
    "binary_lng",
    "multi_lng",
    "fit_nrtl",
    "plot_nrtl_fitting",
    "__version__",
]

_PUBLIC_FUNCTIONS = frozenset(__all__) - {"__version__"}


def __getattr__(name: str):
    if name not in _PUBLIC_FUNCTIONS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".core", __name__), name)
    globals()[name] = value
    return value
```

Extend `.gitignore` with `.pytest_cache/`, `.coverage`, `build/`, `dist/`, `*.egg-info/`, and generated `src/tennetsac/_version.py`.

- [ ] **Step 5: Install editable metadata and run the tests**

Run: `python -m pip install --no-deps -e . && python -m pytest tests/test_package_metadata.py -v`

Expected: both tests pass; the reported development version is PEP 440 compliant and derived from the current Git state.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/tennetsac/__init__.py tests/test_package_metadata.py
git commit -m "build: derive package version from git"
```

---

### Task 2: Migrate the package sources and unchanged resources

**Files:**
- Move: `models/` to `src/tennetsac/models/`
- Move: `smi_ted_light/` to `src/tennetsac/smi_ted_light/`
- Move: `ckpt_files/` to `src/tennetsac/ckpt_files/`
- Move: `utils/` to `src/tennetsac/utils/`
- Create: `src/tennetsac/models/__init__.py`
- Create: `src/tennetsac/utils/__init__.py`
- Create: `src/tennetsac/smi_ted_light/__init__.py`
- Create: `src/tennetsac/core.py`
- Test: `tests/test_source_migration.py`
- Modify: `tests/test_smi_ted_loading.py`
- Modify: `tests/test_smi_ted_tokenizer.py`

**Interfaces:**
- Consumes: the published implementation in `../../pypi/TSAC_pypi/tennetsac/` and the fixed GitHub SMI-TED code at commit `df401e4`.
- Produces: one importable package tree under `src/tennetsac`; no root-level duplicate Python modules or checkpoints.

- [ ] **Step 1: Write failing migration tests**

Add `tests/test_source_migration.py`:

```python
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_only_src_tree_contains_runtime_sources():
    for old_path in ("models", "utils", "smi_ted_light", "ckpt_files"):
        assert not (ROOT / old_path).exists()


def test_package_tree_contains_required_sources():
    package = ROOT / "src" / "tennetsac"
    expected = {
        "core.py",
        "models/Emb2Geometry.py",
        "models/Emb2Profile.py",
        "models/Prf2Gamma.py",
        "utils/embedding.py",
        "utils/model_io.py",
        "utils/property.py",
        "utils/smiles.py",
        "smi_ted_light/load.py",
        "smi_ted_light/tokenizer.py",
        "smi_ted_light/bert_vocab_curated.txt",
        "ckpt_files/base.ckpt",
        "ckpt_files/geo.ckpt",
        "ckpt_files/prf.ckpt",
    }
    assert not [path for path in expected if not (package / path).is_file()]
```

- [ ] **Step 2: Run the migration tests and confirm they fail**

Run: `python -m pytest tests/test_source_migration.py -v`

Expected: both tests fail because runtime sources remain at repository root.

- [ ] **Step 3: Move GitHub-owned sources and binary resources**

Use `git mv` so checkpoint identity and file history remain visible:

```bash
git mv models src/tennetsac/models
git mv smi_ted_light src/tennetsac/smi_ted_light
git mv ckpt_files src/tennetsac/ckpt_files
git mv utils src/tennetsac/utils
```

Add empty `__init__.py` files to `models`, `utils`, and `smi_ted_light`. Do not copy the stale `.so`, `.pyc`, `__pycache__`, `egg-info`, or `dist` files from `TSAC_pypi`.

- [ ] **Step 4: Restore the published behavioral baseline**

Create `src/tennetsac/core.py` from `../../pypi/TSAC_pypi/tennetsac/core.py`. Replace package imports so they remain relative. Replace `src/tennetsac/utils/property.py` with the 161-line published PyPI file, not the GitHub mean/std ensemble variant. Adapt the published embedding imports to the final sibling layout:

```python
from transformers import RobertaModel, RobertaTokenizer

from ..smi_ted_light.load import load_smi_ted
from .smiles import canonicalize_smiles
```

Retain the `src/tennetsac/smi_ted_light/tokenizer.py` and `load.py` already present from commit `df401e4`; those two files override the old PyPI copies.

- [ ] **Step 5: Update SMI-TED tests to final imports**

Use these imports and vocabulary location:

```python
from tennetsac.smi_ted_light import load as smi_ted_load
from tennetsac.smi_ted_light.tokenizer import MolTranBertTokenizer

VOCAB_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "tennetsac"
    / "smi_ted_light"
    / "bert_vocab_curated.txt"
)
```

- [ ] **Step 6: Verify source selection and checkpoint identity**

Run:

```bash
python -m pytest tests/test_source_migration.py tests/test_smi_ted_tokenizer.py tests/test_smi_ted_loading.py -v
shasum -a 256 src/tennetsac/ckpt_files/base.ckpt src/tennetsac/ckpt_files/geo.ckpt src/tennetsac/ckpt_files/prf.ckpt src/tennetsac/ckpt_files/fine-tuned/*.ckpt
```

Expected: tests pass, and hashes equal the values introduced in Task 5. Ensure `src/tennetsac/utils/property.py` returns only ensemble means, matching PyPI 0.1.10.

- [ ] **Step 7: Commit**

```bash
git add src tests/test_source_migration.py tests/test_smi_ted_loading.py tests/test_smi_ted_tokenizer.py
git commit -m "refactor: consolidate package under src"
```

---

### Task 3: Introduce lazy, cached runtime initialization

**Files:**
- Create: `src/tennetsac/runtime.py`
- Modify: `src/tennetsac/core.py`
- Modify: `src/tennetsac/__init__.py`
- Test: `tests/test_runtime.py`
- Modify: `tests/test_package_metadata.py`

**Interfaces:**
- Consumes: `load_model(model, checkpoint, device="cpu")`, `load_all_Gamma_models(model_class, checkpoint_dir, num_models=10)`, and package resources under `tennetsac/ckpt_files`.
- Produces: `Runtime`, `_build_runtime() -> Runtime`, and `get_runtime() -> Runtime`, where `get_runtime` is an `functools.lru_cache(maxsize=1)` singleton factory.

- [ ] **Step 1: Write failing runtime tests**

Use monkeypatching so the tests never instantiate real neural models:

```python
def test_get_runtime_builds_once(monkeypatch):
    from tennetsac import runtime

    marker = object()
    calls = []
    runtime.get_runtime.cache_clear()
    monkeypatch.setattr(runtime, "_build_runtime", lambda: calls.append(1) or marker)

    assert runtime.get_runtime() is marker
    assert runtime.get_runtime() is marker
    assert calls == [1]


def test_plain_import_cannot_trigger_runtime(monkeypatch):
    import importlib
    import tennetsac.runtime as runtime

    runtime.get_runtime.cache_clear()
    monkeypatch.setattr(
        runtime,
        "_build_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("runtime initialized")),
    )
    importlib.reload(importlib.import_module("tennetsac"))
```

Also test that a missing resource raises `FileNotFoundError` containing the logical checkpoint name and `tennetsac/ckpt_files`.

- [ ] **Step 2: Run the runtime tests and confirm missing interfaces**

Run: `python -m pytest tests/test_runtime.py -v`

Expected: collection fails because `tennetsac.runtime` does not exist.

- [ ] **Step 3: Implement the runtime holder**

Use a frozen dataclass and build all heavyweight objects in one function:

```python
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files


@dataclass(frozen=True)
class Runtime:
    chemberta_embedder: object
    smi_ted_embedder: object
    profile_model: object
    geometry_model: object
    gamma_base_model: object
    gamma_finetuned_models: tuple[object, ...]


def _checkpoint_root():
    root = files("tennetsac").joinpath("ckpt_files")
    required = ("base.ckpt", "geo.ckpt", "prf.ckpt")
    missing = [name for name in required if not root.joinpath(name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing packaged checkpoint(s) {missing} under tennetsac/ckpt_files"
        )
    return root


def _build_runtime() -> Runtime:
    from .models.Emb2Geometry import GeometryGenerator
    from .models.Emb2Profile import SigmaProfileGenerator
    from .models.Prf2Gamma import Prf_to_Seg_Model
    from .utils.embedding import ChemBERTaEmbedder, SMITEDEmbedder
    from .utils.model_io import load_all_Gamma_models, load_model

    with as_file(_checkpoint_root()) as checkpoint_root:
        return Runtime(
            chemberta_embedder=ChemBERTaEmbedder(),
            smi_ted_embedder=SMITEDEmbedder(),
            profile_model=load_model(SigmaProfileGenerator(), checkpoint_root / "prf.ckpt"),
            geometry_model=load_model(GeometryGenerator(), checkpoint_root / "geo.ckpt"),
            gamma_base_model=load_model(Prf_to_Seg_Model(), checkpoint_root / "base.ckpt"),
            gamma_finetuned_models=tuple(
                load_all_Gamma_models(
                    Prf_to_Seg_Model, checkpoint_root / "fine-tuned"
                )
            ),
        )


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return _build_runtime()
```

Keep imports for torch, transformers, RDKit, model architectures, and embedding classes inside `_build_runtime()` wherever possible.

- [ ] **Step 4: Rewire the public core**

Delete all module-level `ChemBERTaEmbedder()`, `SMITEDEmbedder()`, and checkpoint-loading statements. Remove `core.py` imports of model architectures, model loaders, embedding classes, Transformers logging, and global warning-filter mutations; runtime construction owns those concerns. Resolve the cached runtime inside the existing private wrappers:

```python
def sigma_profile_wrapper(smiles):
    runtime = get_runtime()
    return get_sigma_profile(
        smiles,
        runtime.profile_model,
        runtime.geometry_model,
        runtime.chemberta_embedder,
        runtime.smi_ted_embedder,
    )


def single_model_predictor(sigma, temperature):
    model = get_runtime().gamma_base_model
    return model(sigma, torch.tensor([temperature]))[1]


def ensemble_predictor(sigma, temperature):
    return ensemble_segac(get_runtime().gamma_finetuned_models, sigma, temperature)
```

Import `least_squares` inside `fit_nrtl` and `matplotlib.pyplot` inside `plot_nrtl_fitting` so resolving another API does not load optional behavior. Keep all published signatures and conversion to Python lists unchanged.

- [ ] **Step 5: Verify lazy import and caching**

Run:

```bash
python -m pytest tests/test_runtime.py tests/test_package_metadata.py -v
python -c 'import sys, tennetsac; assert "transformers" not in sys.modules; assert "torch" not in sys.modules'
```

Expected: all tests and the subprocess assertion pass without network access.

- [ ] **Step 6: Commit**

```bash
git add src/tennetsac/__init__.py src/tennetsac/core.py src/tennetsac/runtime.py tests/test_runtime.py tests/test_package_metadata.py
git commit -m "refactor: load model runtime on first use"
```

---

### Task 4: Lock the published API contract

**Files:**
- Create: `tests/test_api_contract.py`
- Modify: `src/tennetsac/core.py`
- Modify: `src/tennetsac/__init__.py`

**Interfaces:**
- Consumes: `get_runtime()` and the existing calculation functions.
- Produces: unchanged public signatures and return types for `profile`, `binary_lng`, `multi_lng`, `fit_nrtl`, and `plot_nrtl_fitting`.

- [ ] **Step 1: Write API signature and behavior tests**

Assert exact signatures:

```python
EXPECTED = {
    "profile": "(smiles: str) -> Tuple[List[float], float, float]",
    "binary_lng": "(smiles: List[str], temperature: float, molefraction: List[float], version: str = 'tuned') -> Tuple[List[float], List[float]]",
    "multi_lng": "(smiles: List[str], temperature: float, composition: List[float], version: str = 'tuned') -> List[float]",
    "fit_nrtl": "(smiles1, smiles2, alpha=0.3, temp_range=None, x_points=21)",
    "plot_nrtl_fitting": "(smiles1, smiles2, fit_result)",
}


def test_prediction_signatures_are_stable():
    import inspect
    import tennetsac

    for name, expected in EXPECTED.items():
        assert str(inspect.signature(getattr(tennetsac, name))) == expected
```

Monkeypatch `core.sigma_profile_wrapper`, `core.calc_ln_gamma_binary`, and `core.calc_ln_gamma` to return small tensors/arrays. Assert:

- `profile("CCO")` returns `(list[float], float, float)`.
- `binary_lng(...)` returns two Python lists.
- `multi_lng(...)` returns one Python list.
- a non-two-element binary SMILES list raises the existing `ValueError`.
- `version="invalid"` raises the existing model-selection `ValueError`.

- [ ] **Step 2: Run tests and capture any migration regressions**

Run: `python -m pytest tests/test_api_contract.py -v`

Expected: failures identify signature, import, or return-conversion differences from PyPI 0.1.10.

- [ ] **Step 3: Make only compatibility corrections**

Correct annotations, default values, error messages, and `.tolist()` conversions to match the published baseline. Do not add ensemble standard deviations or new function arguments in this task.

- [ ] **Step 4: Run focused and SMI-TED regression suites**

Run: `python -m pytest tests/test_api_contract.py tests/test_runtime.py tests/test_smi_ted_tokenizer.py tests/test_smi_ted_loading.py -v`

Expected: all tests pass without downloading a model.

- [ ] **Step 5: Commit**

```bash
git add src/tennetsac/core.py src/tennetsac/__init__.py tests/test_api_contract.py
git commit -m "test: lock published package api"
```

---

### Task 5: Add the model manifest and pinned external revisions

**Files:**
- Create: `src/tennetsac/model_manifest.json`
- Create: `src/tennetsac/model_manifest.py`
- Create: `src/tennetsac/assets/README.md`
- Create: `tests/test_model_manifest.py`
- Create: `tests/test_embeddings.py`
- Modify: `src/tennetsac/runtime.py`
- Modify: `src/tennetsac/utils/embedding.py`
- Modify: `src/tennetsac/smi_ted_light/load.py`
- Modify: `tests/test_smi_ted_loading.py`

**Interfaces:**
- Consumes: package resources and `hashlib.sha256`.
- Produces: `load_manifest() -> dict`, `external_model(name: str) -> dict`, `verify_bundled_artifacts() -> list[str]`; embedder constructors accept manifest-derived `source` and `revision` values.

- [ ] **Step 1: Write failing manifest tests**

Tests must assert:

```python
EXPECTED_ARTIFACTS = {
    "gamma-base": ("ckpt_files/base.ckpt", "791ce5bf9c59e2099467882b4e7ee220575f464f2f3e007495a78ed9ae6f2d93"),
    "geometry": ("ckpt_files/geo.ckpt", "ab4e37731eb07573cf4d9a7a15c906c6ef8b0ae5a895ff4f921a736c32483624"),
    "sigma-profile": ("ckpt_files/prf.ckpt", "649e7139cc43a95bd459d0eb0c58fac3a5d3a62a9b15ae6bdb0018894c42723e"),
    "gamma-tuned-1": ("ckpt_files/fine-tuned/1.ckpt", "134be46cce6839a7b08250750e4803c8198a4725d5973ca124db76dd488aedc1"),
    "gamma-tuned-2": ("ckpt_files/fine-tuned/2.ckpt", "d763688bbe2898fc182ae6e9b39436a678ad892510c76b510c43c389e87d4c5b"),
    "gamma-tuned-3": ("ckpt_files/fine-tuned/3.ckpt", "15232df78dbbce274818e00bdfff455a163fbb9ff8af1ef3ba4de73650519400"),
    "gamma-tuned-4": ("ckpt_files/fine-tuned/4.ckpt", "bf01d2c6832b161ddf12d789dc886a3c3065a1810948d196f103a89c94f0daeb"),
    "gamma-tuned-5": ("ckpt_files/fine-tuned/5.ckpt", "937a02283521bba254b917685f4db265c48617732f9c5bbfffb14eaaf1d38813"),
    "gamma-tuned-6": ("ckpt_files/fine-tuned/6.ckpt", "7588bfe7a265752395373cf930ede2e721da2cb37fd7779a7b3b94d6c3590e79"),
    "gamma-tuned-7": ("ckpt_files/fine-tuned/7.ckpt", "f4c03f8281b7ac56213e5e04123c208fc4f2030ca610684bd1cbd9aa59f7e16d"),
    "gamma-tuned-8": ("ckpt_files/fine-tuned/8.ckpt", "0ff319a81281b4315638281283433590e6adf7b7ab17039919601747f736096d"),
    "gamma-tuned-9": ("ckpt_files/fine-tuned/9.ckpt", "73c65639135717a3eaa7130c03ea6f85fa3f95ed1b5937b5b73611bd33378c70"),
    "gamma-tuned-10": ("ckpt_files/fine-tuned/10.ckpt", "def4bb97274011564b33d55666bcc5d4d2bcae9c422cbf2c96529c7c364ebfcc"),
}
```

Assert `schema_version == 1`, `bundle_version == "1.0.0"`, the artifact set equals this mapping, every digest is lowercase hexadecimal of length 64, and `verify_bundled_artifacts() == []`.

Assert external entries contain:

```python
{
    "chemberta2": {
        "source": "DeepChem/ChemBERTa-77M-MLM",
        "revision": "ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
        "distribution": "external",
    },
    "smi-ted-light": {
        "source": "ibm/materials.smi-ted",
        "filename": "smi-ted-Light_40.pt",
        "revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
        "sha256": "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
        "distribution": "external",
    },
}
```

- [ ] **Step 2: Run manifest tests and confirm missing resources**

Run: `python -m pytest tests/test_model_manifest.py -v`

Expected: collection fails because `tennetsac.model_manifest` does not exist.

- [ ] **Step 3: Implement manifest loading and verification**

Create a package-resource loader and verifier:

```python
import hashlib
import json
from importlib.resources import as_file, files


def load_manifest() -> dict:
    resource = files("tennetsac").joinpath("model_manifest.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def external_model(name: str) -> dict:
    for entry in load_manifest()["external_models"]:
        if entry["name"] == name:
            return entry
    raise KeyError(f"Unknown external model: {name}")


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundled_artifacts() -> list[str]:
    errors = []
    package_root = files("tennetsac")
    for entry in load_manifest()["artifacts"]:
        resource = package_root.joinpath(entry["path"])
        if not resource.is_file():
            errors.append(f"missing: {entry['path']}")
            continue
        with as_file(resource) as path:
            digest = _sha256(path)
        if digest != entry["sha256"]:
            errors.append(f"sha256 mismatch: {entry['path']}")
    return errors
```

Create the JSON using the exact hashes and revisions above. Add tokenizer metadata for `smi-ted-regex`, format version `1`, and `smi_ted_light/bert_vocab_curated.txt`.

- [ ] **Step 4: Test revision forwarding before implementation**

Mock `RobertaTokenizer.from_pretrained`, `RobertaModel.from_pretrained`, and `hf_hub_download`. Assert both ChemBERTa calls receive the manifest revision and SMI-TED fallback receives exactly:

```python
{
    "repo_id": "ibm/materials.smi-ted",
    "filename": "smi-ted-Light_40.pt",
    "revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
}
```

Create a deliberately wrong local SMI-TED file and assert the manifest-driven runtime path rejects it before `torch.load`, with an error that names `smi-ted-Light_40.pt`, expected SHA256, and actual SHA256.

Run: `python -m pytest tests/test_embeddings.py tests/test_smi_ted_loading.py -v`

Expected: tests fail because current constructors and `load_smi_ted` do not accept or forward revisions.

- [ ] **Step 5: Wire manifest entries into model initialization**

Use explicit keyword interfaces:

```python
class ChemBERTaEmbedder:
    def __init__(self, model_name, revision, max_length=128, device="cpu"):
        self.tokenizer = RobertaTokenizer.from_pretrained(
            model_name, revision=revision
        )
        self.model = RobertaModel.from_pretrained(
            model_name, revision=revision
        ).to(device)


class SMITEDEmbedder:
    def __init__(self, model_dir, repo_id, revision, ckpt_name, expected_sha256, device="cpu"):
        self.model = load_smi_ted(
            folder=model_dir,
            repo_id=repo_id,
            revision=revision,
            ckpt_filename=ckpt_name,
            expected_sha256=expected_sha256,
        ).to(device)
```

Extend `load_smi_ted` with `repo_id`, `revision`, and optional `expected_sha256`. Pass the first two to `hf_hub_download`, stream-hash the selected local or downloaded file when a digest is supplied, and raise `ValueError` before `torch.load` on mismatch. In `_build_runtime`, fetch `external_model("chemberta2")` and `external_model("smi-ted-light")`; supply their exact values, including the SMI digest. Resolve the packaged vocabulary directory locally, but allow the SMI checkpoint to fall back to its pinned Hugging Face revision. If `TENNETSAC_SMI_TED_CHECKPOINT` is set, require it to name a file and pass its parent directory and filename to `SMITEDEmbedder`; the same manifest digest still applies. Wrap initialization failures with a model-specific message and preserve the original exception through `raise ... from error`.

- [ ] **Step 6: Document the reserved ChemBERTa2 asset directory**

`src/tennetsac/assets/README.md` must state that `assets/chemberta2/` is reserved for a later offline bundle, that no files are bundled in this release, and that redistribution terms must be reviewed before addition.

- [ ] **Step 7: Run hash and forwarding tests**

Run: `python -m pytest tests/test_model_manifest.py tests/test_embeddings.py tests/test_smi_ted_loading.py -v`

Expected: all pass; no test performs a real download.

- [ ] **Step 8: Commit**

```bash
git add src/tennetsac/model_manifest.json src/tennetsac/model_manifest.py src/tennetsac/assets src/tennetsac/runtime.py src/tennetsac/utils/embedding.py src/tennetsac/smi_ted_light/load.py tests/test_model_manifest.py tests/test_embeddings.py tests/test_smi_ted_loading.py
git commit -m "feat: pin model bundle provenance"
```

---

### Task 6: Verify built distributions and optional full-model behavior

**Files:**
- Create: `MANIFEST.in`
- Create: `scripts/verify_distribution.py`
- Create: `tests/test_distribution.py`
- Create: `tests/fixtures/pypi_0_1_10_outputs.json`
- Create: `tests/integration/test_full_model.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: wheel and sdist paths in `dist/`; optional `TENNETSAC_SMI_TED_CHECKPOINT` environment variable.
- Produces: `verify_archive(path: Path) -> list[str]`, a nonzero verification CLI on invalid artifacts, and opt-in integration coverage.

- [ ] **Step 1: Write failing archive-verifier tests**

Create small synthetic ZIP and tar archives. Verify rejection of members containing any of:

```python
FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "tennetsac.egg-info",
    "dist",
}
FORBIDDEN_SUFFIXES = {".pyc", ".so"}
```

For wheels, also reject top-level repository directories `tests`, `docs`, `.github`, `examples`, and `scripts`. For sdists, allow the required root build files but reject those same repository-only directories plus `TeNNet-SAC.yml`, `requirements.txt`, `TrainingSystemsList.csv`, `CITATION.bib`, `CITATION.ris`, and `architecture.png`. Verify acceptance requires `tennetsac/model_manifest.json`, the SMI-TED vocabulary, all 13 checkpoint files, and package metadata. Verify `main([archive])` returns `1` and prints every violation when errors exist.

- [ ] **Step 2: Run verifier tests and confirm the script is absent**

Run: `python -m pytest tests/test_distribution.py -v`

Expected: collection fails because `scripts.verify_distribution` does not exist.

- [ ] **Step 3: Implement wheel and sdist inspection**

Use `zipfile.ZipFile` for wheels and `tarfile.open(..., "r:*")` for sdists. Normalize member paths by stripping the single sdist top-level directory before applying the required/forbidden rules. Expose a CLI:

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+")
    args = parser.parse_args(argv)
    errors = [
        f"{archive}: {error}"
        for archive in map(Path, args.archives)
        for error in verify_archive(archive)
    ]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0
```

Create `MANIFEST.in` so the sdist contains only its build inputs and package tree:

```text
include LICENSE README.md pyproject.toml
graft src
prune .github
prune docs
prune examples
prune scripts
prune tests
exclude CITATION.bib CITATION.ris
exclude TeNNet-SAC.yml requirements.txt TrainingSystemsList.csv architecture.png
global-exclude *.py[cod] *.so
prune src/**/__pycache__
```

- [ ] **Step 4: Add the opt-in full-model test**

Mark the test `integration`. Skip unless `TENNETSAC_SMI_TED_CHECKPOINT` names an existing file and ChemBERTa2 is already present in the Hugging Face cache. Commit this fixture, measured offline from local PyPI 0.1.10 with Python 3.10.19, PyTorch 2.1.2, and Transformers 4.36.2:

```json
{
  "smiles": "CCO",
  "sigma": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30202516913414, 0.6570189595222473, 0.8024698495864868, 0.7677388787269592, 0.7602039575576782, 0.8310112953186035, 0.8506041765213013, 0.7559620141983032, 0.7572101950645447, 0.7031354904174805, 1.0797531604766846, 4.483050346374512, 9.191164016723633, 9.759830474853516, 8.870887756347656, 8.263846397399902, 7.464892864227295, 7.461089134216309, 6.359116554260254, 2.096234083175659, 1.5049933195114136, 1.2838516235351562, 0.9547985196113586, 1.1031712293624878, 1.0997263193130493, 1.0706664323806763, 1.157448410987854, 1.150734543800354, 1.2214977741241455, 1.5971901416778564, 2.2041525840759277, 1.9979612827301025, 0.8095311522483826, 0.14849939942359924, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "area": 89.5214614868164,
  "volume": 70.52980041503906,
  "binary": {
    "smiles": ["CCO", "ClCCCl"],
    "temperature": 298.15,
    "molefraction": [0.25, 0.5, 0.75],
    "lng1": [0.932470578586941, 0.3458164570181846, 0.06820888056001345],
    "lng2": [0.10410286995124578, 0.45153779862256593, 0.9037106524924411]
  },
  "multi": {
    "smiles": ["CCO", "ClCCCl", "CCN"],
    "temperature": 298.15,
    "composition": [0.3, 0.4],
    "lng": [0.31023567453225104, 0.380520729077282, -0.23821974109875502]
  }
}
```

With both offline environment variables set, test the fixed inputs:

```python
SMILES = ["CCO", "ClCCCl"]


def test_full_runtime_outputs_are_finite():
    from tennetsac import binary_lng, profile

    sigma, area, volume = profile("CCO")
    lng1, lng2 = binary_lng(SMILES, 298.15, [0.25, 0.5, 0.75])
    assert len(sigma) == 51
    assert area > 0 and volume > 0
    assert np.isfinite([*sigma, *lng1, *lng2]).all()
```

Load the JSON fixture and compare sigma, area, volume, binary outputs, and multicomponent outputs with `numpy.testing.assert_allclose(..., rtol=1e-5, atol=1e-6)`. The fixture contains outputs only; do not commit model caches.

- [ ] **Step 5: Build and inspect real artifacts**

Run:

```bash
python -m build
python -m twine check dist/*
python scripts/verify_distribution.py dist/*
```

Expected: one wheel and one sdist pass metadata and content checks; neither contains forbidden cache/build files or the SMI-TED checkpoint.

- [ ] **Step 6: Install the wheel outside the repository**

Create a temporary virtual environment, install the wheel with runtime dependencies already satisfied or `--no-deps`, change the working directory to a temporary directory, and run:

```bash
python -c 'import importlib.resources as r, tennetsac; assert r.files("tennetsac").joinpath("ckpt_files/base.ckpt").is_file(); print(tennetsac.__version__)'
```

Expected: import succeeds without network access and the packaged checkpoint is available.

- [ ] **Step 7: Commit**

```bash
git add MANIFEST.in scripts/verify_distribution.py tests/test_distribution.py tests/fixtures/pypi_0_1_10_outputs.json tests/integration/test_full_model.py pyproject.toml
git commit -m "test: verify installable release artifacts"
```

---

### Task 7: Update examples and repository documentation

**Files:**
- Move: `TeNNetSAC.ipynb` to `examples/TeNNetSAC.ipynb`
- Modify: `examples/TeNNetSAC.ipynb`
- Modify: `README.md`
- Modify: `requirements.txt`
- Modify: `TeNNet-SAC.yml`
- Test: `tests/test_documentation.py`

**Interfaces:**
- Consumes: the installable `tennetsac` public API and `pyproject.toml` dependency policy.
- Produces: one documented installation path and examples without repository-root imports.

- [ ] **Step 1: Write documentation consistency tests**

Parse README and notebook JSON and assert:

```python
assert "from tennetsac import" in readme
assert "examples/TeNNetSAC.ipynb" in readme
assert "from core import" not in notebook_source
assert "from utils" not in notebook_source
assert "from models" not in notebook_source
```

Also assert `requirements.txt` contains `-e .` plus development-only tools rather than duplicating runtime version constraints maintained in `pyproject.toml`.

- [ ] **Step 2: Run documentation tests and confirm old paths fail**

Run: `python -m pytest tests/test_documentation.py -v`

Expected: failures reference the root notebook and old project-structure table.

- [ ] **Step 3: Move and normalize the notebook**

Use `git mv TeNNetSAC.ipynb examples/TeNNetSAC.ipynb`. Replace internal imports with:

```python
from tennetsac import binary_lng, multi_lng, profile
```

Do not add `sys.path` manipulation or source-tree-relative checkpoint paths.

- [ ] **Step 4: Update README and development dependencies**

Document:

- PyPI installation with `pip install tennetsac`.
- source development with `pip install -e ".[dev]"`.
- model initialization occurs on the first prediction call, not import.
- ChemBERTa2 and SMI-TED are revision-pinned external downloads for this release.
- the future offline ChemBERTa2 location is reserved but currently empty.
- versions come from Git tags and the GitHub repository is the release source.
- the notebook path and `src/tennetsac` project layout.

Set `requirements.txt` to:

```text
-e .[dev]
```

Update the conda environment to install the local package through pip rather than restating PyPI dependencies. Preserve the existing CPU/GPU environment intent.

- [ ] **Step 5: Run documentation checks**

Run: `python -m pytest tests/test_documentation.py -v`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add README.md requirements.txt TeNNet-SAC.yml examples tests/test_documentation.py
git commit -m "docs: document unified package workflow"
```

---

### Task 8: Add continuous integration

**Files:**
- Create: `.github/workflows/ci.yml`
- Test: `tests/test_workflows.py`

**Interfaces:**
- Consumes: extras `.[test,build]`, pytest marker `integration`, and `scripts/verify_distribution.py`.
- Produces: required checks for Python 3.10, 3.11, and 3.12 plus isolated tokenizer compatibility for Transformers 4.36.2 and latest v5.

- [ ] **Step 1: Write failing workflow structure tests**

Load `.github/workflows/ci.yml` with `yaml.load(..., Loader=yaml.BaseLoader)` so the YAML 1.1 parser does not coerce the key `on` to a Boolean. Assert:

- triggers include pull requests, pushes to `main`, and `workflow_call` for tag-release reuse;
- permissions are `contents: read`;
- unit test matrix is exactly `3.10`, `3.11`, `3.12`;
- normal tests exclude `integration`;
- a build job runs `python -m build`, `twine check`, and `scripts/verify_distribution.py`;
- the build job installs and imports the wheel from outside the checkout;
- tokenizer matrix includes `transformers==4.36.2` and `transformers>=5,<6`;
- tokenizer job installs the package with `--no-deps` and runs only `tests/test_smi_ted_tokenizer.py`.

- [ ] **Step 2: Run workflow tests and confirm the file is absent**

Run: `python -m pytest tests/test_workflows.py -v`

Expected: failure because `.github/workflows/ci.yml` does not exist.

- [ ] **Step 3: Implement the CI workflow**

Create jobs `unit`, `tokenizer-compatibility`, and `distribution`. Use `actions/checkout`, `actions/setup-python`, and pip caching. Expose the workflow through `workflow_call` in addition to pull-request and main-push triggers. Set these environment values at workflow or job level:

```yaml
env:
  HF_HUB_OFFLINE: "1"
  TRANSFORMERS_OFFLINE: "1"
  PYTHONWARNINGS: "error::ResourceWarning"
```

The normal unit command is:

```bash
python -m pytest -m "not integration" -v
```

The installed-wheel smoke check must change to a temporary directory before importing so `src/` cannot shadow the wheel.

- [ ] **Step 4: Validate YAML and run the workflow contract tests**

Run: `python -m pytest tests/test_workflows.py -v`

Expected: all workflow structure tests pass.

- [ ] **Step 5: Run the same local gates as CI**

Run:

```bash
python -m pytest -m "not integration" -v
python -m build
python -m twine check dist/*
python scripts/verify_distribution.py dist/*
```

Expected: all commands exit zero.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml tests/test_workflows.py
git commit -m "ci: test unified Python package"
```

---

### Task 9: Add draft release construction and protected PyPI publication

**Files:**
- Create: `.github/workflows/release-build.yml`
- Create: `.github/workflows/publish-pypi.yml`
- Create: `scripts/verify_release_version.py`
- Modify: `tests/test_workflows.py`
- Create: `tests/test_release_version.py`

**Interfaces:**
- Consumes: a `v<version>` Git tag, built `dist/*.whl` and `dist/*.tar.gz`, and a protected GitHub environment named `pypi`.
- Produces: a draft GitHub Release with checksummed artifacts and a manual trusted-publishing workflow that downloads those exact assets.

- [ ] **Step 1: Write failing release-version tests**

Test normalization and mismatch behavior:

```python
@pytest.mark.parametrize(
    ("tag", "expected"),
    [("v0.2.0", "0.2.0"), ("v1.2.3rc1", "1.2.3rc1")],
)
def test_version_from_tag(tag, expected):
    assert version_from_tag(tag) == expected


@pytest.mark.parametrize("tag", ["0.2.0", "release-v1", "vnext", "v1.0+local"])
def test_invalid_release_tag_is_rejected(tag):
    with pytest.raises(ValueError):
        version_from_tag(tag)
```

Create a synthetic wheel METADATA file and assert `verify_release_version(tag, wheel)` rejects any version mismatch.

- [ ] **Step 2: Extend workflow tests with release safety invariants**

Assert `release-build.yml`:

- triggers only on tags matching `v*`;
- has `contents: write` and no `id-token: write`;
- calls the reusable `ci.yml`, waits for it to pass, builds exactly once, verifies versions and manifest hashes;
- generates `SHA256SUMS`;
- creates a draft GitHub Release.

Assert `publish-pypi.yml`:

- triggers only through `workflow_dispatch` with required `tag` input;
- uses environment `pypi` and permissions `id-token: write`, `contents: read`;
- downloads release assets with `gh release download`;
- verifies `SHA256SUMS`, release version, and `twine check`;
- contains no `python -m build`, `build`, `sdist`, or `bdist` step;
- publishes with `pypa/gh-action-pypi-publish`.

- [ ] **Step 3: Run release tests and confirm missing implementation**

Run: `python -m pytest tests/test_release_version.py tests/test_workflows.py -v`

Expected: failures for the missing verifier and release workflows.

- [ ] **Step 4: Implement strict tag/artifact verification**

`version_from_tag` must remove one leading `v`, parse with `packaging.version.Version`, reject local versions, and require the canonical string to equal the remainder. `verify_release_version` reads wheel metadata through `zipfile` and sdist `PKG-INFO` through `tarfile`; it reports a mismatch unless every artifact version equals the normalized tag.

CLI usage:

```bash
python scripts/verify_release_version.py --tag "$GITHUB_REF_NAME" dist/*
```

- [ ] **Step 5: Implement the tag build workflow**

Define a first job that reuses `./.github/workflows/ci.yml`. Make the release job depend on that reusable workflow. The release job sequence is:

```yaml
- run: python -m pytest -m "not integration" -v
- run: python -m build
- run: python -m twine check dist/*
- run: python scripts/verify_distribution.py dist/*
- run: python -c 'from tennetsac.model_manifest import verify_bundled_artifacts; errors = verify_bundled_artifacts(); assert not errors, errors'
- run: python scripts/verify_release_version.py --tag "$GITHUB_REF_NAME" dist/*
- run: python -m pip install --force-reinstall --no-deps dist/*.whl
- run: python -c 'import os, tennetsac; assert tennetsac.__version__ == os.environ["GITHUB_REF_NAME"].removeprefix("v")'
- run: cd dist && shasum -a 256 * > SHA256SUMS
```

Create the draft release with `gh release create "$GITHUB_REF_NAME" dist/*.whl dist/*.tar.gz dist/SHA256SUMS --draft --verify-tag` and `GH_TOKEN: ${{ github.token }}`. Do not call a PyPI publisher.

- [ ] **Step 6: Implement manual protected publication**

The publish job must use `environment: pypi`, download exactly the user-supplied existing tag into an empty `dist/`, run `shasum -a 256 -c SHA256SUMS`, validate versions and metadata, and then call:

```yaml
- uses: pypa/gh-action-pypi-publish@release/v1
  with:
    packages-dir: dist/
```

Remove `SHA256SUMS` from `dist/` immediately before the publishing action so only wheel and sdist files are uploaded. Do not publish or undraft the GitHub Release automatically.

- [ ] **Step 7: Run workflow and verifier tests**

Run: `python -m pytest tests/test_release_version.py tests/test_workflows.py -v`

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/release-build.yml .github/workflows/publish-pypi.yml scripts/verify_release_version.py tests/test_release_version.py tests/test_workflows.py
git commit -m "ci: build once and publish releases manually"
```

---

### Task 10: Run the final migration verification

**Files:**
- Modify only files required to fix failures found by the verification commands.

**Interfaces:**
- Consumes: all package, test, documentation, and workflow outputs from Tasks 1-9.
- Produces: a clean feature branch ready for review; no tag, GitHub Release, or PyPI upload.

- [ ] **Step 1: Confirm repository structure and source uniqueness**

Run:

```bash
python -m pytest tests/test_source_migration.py tests/test_documentation.py -v
find . -type d -name __pycache__ -prune -o -type f \( -name 'core.py' -o -name 'Emb2Profile.py' -o -name 'bert_vocab_curated.txt' \) -print
```

Expected: tests pass and each maintained runtime file appears only under `src/tennetsac`.

- [ ] **Step 2: Run the complete offline test suite**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -m "not integration" -v
```

Expected: all tests pass and no model download is attempted.

- [ ] **Step 3: Rebuild artifacts from a clean output directory**

Delete only the generated repository-local `dist/` and `build/` directories after resolving them to the current checkout. Then run:

```bash
python -m build
python -m twine check dist/*
python scripts/verify_distribution.py dist/*
```

Expected: all artifact checks pass.

- [ ] **Step 4: Verify the installed wheel outside the checkout**

Install the wheel into a new temporary environment with `--no-deps`, change to a temporary working directory, set both offline variables, and assert:

```python
import sys
import tennetsac

assert tennetsac.__version__
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
```

Also use `importlib.resources` to confirm all manifest-declared bundled files exist in the wheel.

- [ ] **Step 5: Run the optional full-model equivalence check when caches exist**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TENNETSAC_SMI_TED_CHECKPOINT=/absolute/path/to/smi-ted-Light_40.pt python -m pytest tests/integration/test_full_model.py -v
```

Expected: fixed inputs match the recorded PyPI 0.1.10 golden outputs with `rtol=1e-5` and `atol=1e-6`. If local model files are unavailable, record the test as skipped and do not download 1.15 GB during this gate.

- [ ] **Step 6: Inspect the final diff and commit any verification-only correction**

Run:

```bash
git diff --check
git status --short
git log --oneline main..HEAD
```

If Step 1-5 required a correction, commit only that correction with a message describing the actual fix. Otherwise leave the existing nine focused commits unchanged.

- [ ] **Step 7: Stop before external release actions**

Report the tested wheel/sdist filenames, package version, skipped or completed integration status, and branch commits. Do not push, tag `v0.2.0`, create a GitHub Release, configure PyPI Trusted Publishing, or publish to PyPI without a separate explicit user instruction.
