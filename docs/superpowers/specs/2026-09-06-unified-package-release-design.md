# Unified Package and Release Design

## Context

TeNNet-SAC is currently maintained in two repositories:

- `TeNNet-SAC`, which contains the public GitHub source, notebooks, model
  checkpoints, and recent development changes.
- `TSAC_pypi`, which contains the installable `tennetsac` package, its build
  metadata, and previously built distribution files.

The model checkpoints currently match between the repositories, but the Python
sources do not. The PyPI version is manually fixed at `0.1.10`, neither
repository has a release tag, and there is no workflow that proves a GitHub
revision and a PyPI artifact contain the same code and weights.

The SMI-TED tokenizer compatibility fix in commit `df401e4` is the first change
that must be carried into the unified package.

## Goals

1. Make `TeNNet-SAC` the only maintained source repository.
2. Preserve the existing public package name and API: `import tennetsac`,
   `profile()`, `binary_lng()`, and `multi_lng()`.
3. Derive package versions from Git tags so GitHub and PyPI cannot acquire
   independent version numbers.
4. Build and test wheel and source distributions in CI without downloading
   large model files during import.
5. Record exact model provenance and hashes separately from the package
   version.
6. Produce release artifacts automatically, while requiring a manual,
   protected action before publishing to PyPI.

## Non-goals

- This migration will not implement the shared-trunk ensemble optimization.
- It will not add the ChemBERTa2 model files to the wheel. It will reserve the
  final asset location and manifest fields for that later task.
- It will not bundle the approximately 1.15 GB SMI-TED checkpoint in PyPI.
- It will not adopt the unaligned GitHub ensemble mean/standard-deviation API.
- It will not merge the unrelated Git history of `TSAC_pypi` into this
  repository.

## Repository Architecture

The repository will use a `src` layout:

```text
TeNNet-SAC/
├── .github/workflows/
│   ├── ci.yml
│   ├── release-build.yml
│   └── publish-pypi.yml
├── docs/
├── examples/
│   └── TeNNetSAC.ipynb
├── src/tennetsac/
│   ├── __init__.py
│   ├── _version.py
│   ├── core.py
│   ├── runtime.py
│   ├── model_manifest.json
│   ├── assets/
│   │   └── README.md
│   ├── ckpt_files/
│   │   ├── base.ckpt
│   │   ├── geo.ckpt
│   │   ├── prf.ckpt
│   │   └── fine-tuned/
│   ├── models/
│   ├── smi_ted_light/
│   └── utils/
├── tests/
├── LICENSE
├── README.md
└── pyproject.toml
```

`assets/README.md` documents the reserved
`src/tennetsac/assets/chemberta2/` location. The ChemBERTa2 directory itself is
created only when licensed model files are added, because Git does not preserve
empty directories.

The old root-level `models/`, `utils/`, and `smi_ted_light/` directories will be
removed after their selected contents are migrated. Thin compatibility copies
will not be kept because they would recreate two sources of truth. The example
notebook and README will be updated to import the installed package.

## Source Selection and API Compatibility

The current PyPI package is the behavioral baseline because it defines the
published API. Its `core.py`, package-relative imports, and return formats will
be migrated first. The following rules apply during migration:

1. The SMI-TED implementation from GitHub commit `df401e4` overrides the older
   PyPI SMI-TED tokenizer and loader.
2. Model architecture files are accepted only when their contents match the
   checkpoints and regression tests.
3. GitHub's current ensemble mean/standard-deviation implementation is excluded
   because its function signatures are not aligned with the published
   `core.py`. It will be replaced during the shared-trunk ensemble task.
4. Public function arguments and return types remain unchanged in this
   migration.
5. Existing model checkpoint bytes remain unchanged.

## Runtime Loading

Importing `tennetsac` must not access the network or load neural-network
weights. A private runtime holder will own all heavyweight objects:

```text
public API call
    -> get_runtime() cached singleton
        -> initialize ChemBERTa model/tokenizer
        -> initialize SMI-TED from local path or explicit download fallback
        -> load TeNNet-SAC checkpoints from package resources
    -> execute prediction
```

`tennetsac.__init__` exposes functions and `__version__`, but creates no model
instances. `runtime.get_runtime()` performs first-use initialization and caches
the result for later calls. The ChemBERTa2 model still uses its current remote
identifier during this migration; bundling it under `assets/chemberta2/` is a
later task.

Remote model lookups are revision-pinned from `model_manifest.json` rather than
following a mutable `main` branch. ChemBERTa receives its recorded revision in
`from_pretrained(..., revision=...)`, and the SMI-TED fallback passes its
recorded revision to `hf_hub_download()`. This migration therefore makes remote
resolution reproducible without yet changing how those external files are
distributed.

Package resources will be resolved through `importlib.resources`, not paths
relative to the current working directory. A missing packaged checkpoint raises
a descriptive error naming the expected resource. A missing external SMI-TED
checkpoint either follows the explicit download fallback or reports the local
path and download failure; it never causes network access during package import.

## Versioning

`setuptools-scm` is the only package version source.

- Tag `v0.2.0` produces package version `0.2.0`.
- Untagged development commits produce a PEP 440 development version containing
  commit distance and revision.
- The generated `src/tennetsac/_version.py` is included in wheels and source
  distributions so installed artifacts do not require Git metadata.
- `tennetsac.__version__`, wheel metadata, and the release tag are verified
  against one another in the release workflow.

The first unified release will be `v0.2.0` unless a newer tag already exists at
release time.

## Model Manifest

`src/tennetsac/model_manifest.json` records model identity independently of the
package version. Its initial schema is:

```json
{
  "schema_version": 1,
  "bundle_version": "1.0.0",
  "artifacts": [
    {
      "name": "gamma-base",
      "path": "ckpt_files/base.ckpt",
      "sha256": "64-character lowercase hexadecimal digest",
      "distribution": "bundled"
    }
  ],
  "external_models": [
    {
      "name": "smi-ted-light",
      "source": "ibm/materials.smi-ted",
      "filename": "smi-ted-Light_40.pt",
      "revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
      "sha256": "64-character lowercase hexadecimal digest",
      "distribution": "external"
    }
  ],
  "tokenizers": [
    {
      "name": "smi-ted-regex",
      "format_version": 1,
      "vocab_path": "smi_ted_light/bert_vocab_curated.txt"
    }
  ]
}
```

The actual manifest contains hashes for `base`, `geo`, `prf`, and all ten
fine-tuned checkpoints. ChemBERTa2 receives an `external_models` entry during
this migration and moves to a bundled `artifacts` entry only after its files and
redistribution terms are reviewed in the offline-model task.

Changing code alone increments the package version. Changing weights increments
both the package version and `bundle_version`. CI recomputes all bundled hashes
and fails on any mismatch.

## Build and Release Flow

### Continuous integration

`.github/workflows/ci.yml` runs on pull requests and pushes to `main`:

1. Test supported Python versions 3.10, 3.11, and 3.12.
2. Run unit and regression tests without downloading SMI-TED.
3. Verify package import performs no network or model initialization.
4. Build wheel and source distribution with `python -m build`.
5. Validate metadata with `twine check`.
6. Install the wheel into an isolated environment and run package-level smoke
   tests against the installed artifact rather than the source tree.

The package retains `transformers==4.36.2` during this migration because the
ChemBERTa path has not yet been validated against Transformers v5. A separate,
isolated SMI-TED tokenizer job installs the source without runtime dependency
resolution and runs once with Transformers 4.36.2 and once with the latest v5
release. Both environments must emit the same golden token IDs. Passing that
isolated job does not declare package-level Transformers v5 support.

### Release artifact build

`.github/workflows/release-build.yml` runs for tags matching `v*`:

1. Verify the tag is a valid PEP 440 release after removing the leading `v`.
2. Run the complete CI suite.
3. Build wheel and source distribution exactly once.
4. Verify tag, wheel metadata, and `tennetsac.__version__` match.
5. Verify the model manifest hashes.
6. Upload wheel and source distribution as versioned workflow artifacts and
   attach them to a draft GitHub Release.

This workflow does not publish to PyPI.

### Manual PyPI publication

`.github/workflows/publish-pypi.yml` is manually dispatched for an existing
release tag. It uses a protected GitHub environment and PyPI Trusted Publishing,
downloads the already-built draft GitHub Release assets, verifies their hashes
and versions again, and publishes those exact files. It never rebuilds the
package.

Configuring the PyPI trusted-publisher relationship and approving the protected
environment remain explicit repository-owner actions. A workflow run without
that configuration fails without altering an existing PyPI release.

## Testing Strategy

### Unit and regression tests

- Preserve the SMI-TED golden token IDs for atoms, halogens, stereochemistry,
  charges, and multi-digit ring notation.
- Verify local SMI-TED checkpoint selection and requested fallback filenames.
- Verify model-manifest schema and every bundled SHA256 value.
- Verify package version exposure without importing heavyweight runtime objects.
- Verify the first public API call initializes one runtime and later calls reuse
  it.
- Verify initialization errors name the missing model or resource and preserve
  the underlying cause.

### Package tests

- Build both distribution formats.
- Assert neither artifact contains `__pycache__`, `.pyc`, `egg-info`, old
  `dist/`, or repository-only files.
- Install the wheel outside the repository and import `tennetsac` with network
  access disabled.
- Confirm packaged checkpoint files and tokenizer vocabulary are accessible
  through `importlib.resources`.

### Optional full-model integration

An integration test is enabled only when a local SMI-TED checkpoint path is
provided. It compares fixed SMILES embeddings against the legacy Transformers
4.36.2 implementation and checks finiteness and shape. The 1.15 GB checkpoint is
never downloaded by ordinary CI.

## Migration Sequence

1. Add packaging and version metadata at the repository root.
2. Add tests that describe import, version, resource, and public API behavior.
3. Move the PyPI package into `src/tennetsac` and overlay the SMI-TED fix.
4. Introduce lazy runtime initialization without changing prediction outputs.
5. Generate and validate the model manifest.
6. Move the notebook to `examples/` and update imports and documentation.
7. Remove superseded root-level Python package directories.
8. Add CI, release-build, and manually protected PyPI-publish workflows.
9. Build and install artifacts locally, then run regression and package-content
   checks.
10. After `v0.2.0` is successfully published, archive `TSAC_pypi` and point its
    repository description or README to `TeNNet-SAC`.

## Acceptance Criteria

- `TeNNet-SAC` contains one maintained copy of every package source file.
- `pip install` from the built wheel exposes the unchanged public API.
- Importing `tennetsac` is offline and does not instantiate a model.
- All existing bundled checkpoints have verified manifest hashes.
- SMI-TED tokenizer behavior remains identical in the isolated Transformers
  4.36.2 and latest-v5 compatibility jobs, while package runtime support remains
  pinned to 4.36.2 until ChemBERTa is separately validated.
- A Git tag determines the package version without editing another file.
- CI builds and tests wheel and source distributions.
- Tag builds produce draft GitHub Release artifacts but do not automatically
  publish to PyPI.
- Manual PyPI publication consumes the already-verified artifacts.
- No ChemBERTa2 or SMI-TED licensing assumption is introduced by the migration.
