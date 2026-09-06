# Task 3 report: lazy, cached runtime initialization

## Result

Implemented lazy model runtime initialization. Plain `import tennetsac` remains free of runtime initialization, torch/transformers imports, checkpoint loading, and network access. The first model-backed API call now builds one frozen `Runtime`, and subsequent calls reuse the `functools.lru_cache(maxsize=1)` singleton.

## TDD evidence

### RED

Command:

```bash
python -m pytest tests/test_runtime.py -v
```

Output:

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.11, pytest-8.2.0, pluggy-1.6.0 -- /opt/homebrew/Caskroom/miniforge/base/bin/python
cachedir: .pytest_cache
rootdir: /Users/yueyang/projects/ai_agents/tsac/github/TeNNet-SAC
configfile: pyproject.toml
plugins: anyio-4.13.0
collecting ... collected 3 items

tests/test_runtime.py::test_get_runtime_builds_once FAILED               [ 33%]
tests/test_runtime.py::test_plain_import_cannot_trigger_runtime FAILED   [ 66%]
tests/test_runtime.py::test_missing_checkpoint_reports_logical_package_path FAILED [100%]

=================================== FAILURES ===================================
_________________________ test_get_runtime_builds_once _________________________
ImportError: cannot import name 'runtime' from 'tennetsac' (/Users/yueyang/projects/ai_agents/tsac/github/TeNNet-SAC/src/tennetsac/__init__.py)
___________________ test_plain_import_cannot_trigger_runtime ___________________
ModuleNotFoundError: No module named 'tennetsac.runtime'
_____________ test_missing_checkpoint_reports_logical_package_path _____________
ImportError: cannot import name 'runtime' from 'tennetsac' (/Users/yueyang/projects/ai_agents/tsac/github/TeNNet-SAC/src/tennetsac/__init__.py)

============================== 3 failed in 0.02s ==============================
```

The failures were the expected missing-interface errors: `tennetsac.runtime` did not yet exist.

### GREEN

Command:

```bash
python -m pytest tests/test_runtime.py tests/test_package_metadata.py -v
```

Output:

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.11, pytest-8.2.0, pluggy-1.6.0 -- /opt/homebrew/Caskroom/miniforge/base/bin/python
cachedir: .pytest_cache
rootdir: /Users/yueyang/projects/ai_agents/tsac/github/TeNNet-SAC
configfile: pyproject.toml
plugins: anyio-4.13.0
collecting ... collected 5 items

tests/test_runtime.py::test_get_runtime_builds_once PASSED               [ 20%]
tests/test_runtime.py::test_plain_import_cannot_trigger_runtime PASSED   [ 40%]
tests/test_runtime.py::test_missing_checkpoint_reports_logical_package_path PASSED [ 60%]
tests/test_package_metadata.py::test_package_exposes_pep440_version PASSED [ 80%]
tests/test_package_metadata.py::test_plain_import_does_not_load_model_runtime PASSED [100%]

============================== 5 passed in 0.02s ===============================
```

The import boundary was also checked directly:

```bash
python -c 'import sys, tennetsac; assert "transformers" not in sys.modules; assert "torch" not in sys.modules'
```

This completed successfully with no output.

### Full suite

Command:

```bash
python -m pytest -v
```

Output:

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.11, pytest-8.2.0, pluggy-1.6.0 -- /opt/homebrew/Caskroom/miniforge/base/bin/python
cachedir: .pytest_cache
rootdir: /Users/yueyang/projects/ai_agents/tsac/github/TeNNet-SAC
configfile: pyproject.toml
plugins: anyio-4.13.0
collecting ... collected 16 items

tests/test_package_metadata.py::test_package_exposes_pep440_version PASSED [  6%]
tests/test_package_metadata.py::test_plain_import_does_not_load_model_runtime PASSED [ 12%]
tests/test_runtime.py::test_get_runtime_builds_once PASSED               [ 18%]
tests/test_runtime.py::test_plain_import_cannot_trigger_runtime PASSED   [ 25%]
tests/test_runtime.py::test_missing_checkpoint_reports_logical_package_path PASSED [ 31%]
tests/test_smi_ted_loading.py::test_load_smi_ted_prefers_requested_local_checkpoint PASSED [ 37%]
tests/test_smi_ted_loading.py::test_load_smi_ted_downloads_the_requested_checkpoint_name PASSED [ 43%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_preserves_legacy_smi_ted_tokens_and_ids[CCO-expected_tokens0-expected_ids0] PASSED [ 50%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_preserves_legacy_smi_ted_tokens_and_ids[ClCCCl-expected_tokens1-expected_ids1] PASSED [ 56%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_preserves_legacy_smi_ted_tokens_and_ids[C[C@H](O)C(=O)O-expected_tokens2-expected_ids2] PASSED [ 62%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_preserves_legacy_smi_ted_tokens_and_ids[[NH4+]-expected_tokens3-expected_ids3] PASSED [ 68%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_preserves_legacy_smi_ted_tokens_and_ids[C%12CCCCC%12-expected_tokens4-expected_ids4] PASSED [ 75%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_matches_legacy_batch_padding PASSED [ 81%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_keeps_eos_when_truncating PASSED [ 87%]
tests/test_source_migration.py::test_only_src_tree_contains_runtime_sources PASSED [ 93%]
tests/test_source_migration.py::test_package_tree_contains_required_sources PASSED [100%]

============================== 16 passed in 1.29s ==============================
```

## Files changed

- `src/tennetsac/runtime.py`: added the frozen `Runtime` holder, checkpoint validation, cross-version resource path handling, `_build_runtime`, and cached `get_runtime`.
- `src/tennetsac/core.py`: removed eager embedders/checkpoint/model imports and global warning configuration; wrappers now resolve the cached runtime on use; scipy/matplotlib/property imports are lazy.
- `tests/test_runtime.py`: added cache, import-boundary, and missing-resource tests.
- `tests/test_package_metadata.py`: moved the plain-import assertions into a subprocess so test order cannot contaminate the import-boundary check; now also asserts torch and transformers are absent.

## Compatibility choice

`importlib.resources.as_file` directory support differs across Python versions. The runtime uses unpacked `pathlib.Path` resources directly and, for non-filesystem `Traversable` resources such as zipped wheels, materializes each required checkpoint file into a temporary directory through `as_file`. This avoids relying on Python 3.12-only directory extraction behavior and keeps the existing model loaders, which require filesystem paths/directories, unchanged. The resource directory is temporary only for model loading and is removed after all models are constructed.

## Self-review

- No model architecture, embedding, torch, transformers, scipy, or matplotlib imports remain at `core.py` module scope.
- No module-level embedder construction or checkpoint loading remains.
- Public function signatures and Python-list conversions are unchanged.
- Missing required checkpoints report their logical names and `tennetsac/ckpt_files`.
- `git diff --check` passed.

## Concerns

The test suite intentionally does not build the full neural runtime: doing so would instantiate real models and may require network/model-cache access. The source-tree resource path and Traversable materialization logic are covered structurally; a wheel-install integration run with real weights should be performed in an environment with the model dependencies and caches available.

## Fix Round 1

### Review finding addressed

`_checkpoint_root()` previously validated only the three top-level checkpoints while `_checkpoint_path()` consumed those plus ten fine-tuned checkpoints. The validator now checks all thirteen logical resource paths, including every `fine-tuned/1.ckpt` through `fine-tuned/10.ckpt`, before any extraction or model loading.

### TDD covering tests

Added two focused tests covering the same missing `fine-tuned/10.ckpt` case through separate resource backends:

- `test_missing_finetuned_checkpoint_reports_for_filesystem_resource`
- `test_missing_finetuned_checkpoint_reports_for_traversable_resource`

Both require a descriptive `FileNotFoundError` containing `fine-tuned/10.ckpt` and `tennetsac/ckpt_files`.

RED command:

```bash
python -m pytest tests/test_runtime.py -v
```

RED output:

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.11, pytest-8.2.0, pluggy-1.6.0 -- /opt/homebrew/Caskroom/miniforge/base/bin/python
cachedir: .pytest_cache
rootdir: /Users/yueyang/projects/ai_agents/tsac/github/TeNNet-SAC
configfile: pyproject.toml
plugins: anyio-4.13.0
collecting ... collected 5 items

tests/test_runtime.py::test_get_runtime_builds_once PASSED               [ 20%]
tests/test_runtime.py::test_plain_import_cannot_trigger_runtime PASSED   [ 40%]
tests/test_runtime.py::test_missing_checkpoint_reports_logical_package_path PASSED [ 60%]
tests/test_runtime.py::test_missing_finetuned_checkpoint_reports_for_filesystem_resource FAILED [ 80%]
tests/test_runtime.py::test_missing_finetuned_checkpoint_reports_for_traversable_resource FAILED [100%]

============================== 2 failed, 3 passed in 0.02s ==============================
```

GREEN focused command:

```bash
python -m pytest tests/test_runtime.py tests/test_package_metadata.py -v
```

GREEN output:

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.11, pytest-8.2.0, pluggy-1.6.0 -- /opt/homebrew/Caskroom/miniforge/base/bin/python
cachedir: .pytest_cache
rootdir: /Users/yueyang/projects/ai_agents/tsac/github/TeNNet-SAC
configfile: pyproject.toml
plugins: anyio-4.13.0
collecting ... collected 7 items

tests/test_runtime.py::test_get_runtime_builds_once PASSED               [ 14%]
tests/test_runtime.py::test_plain_import_cannot_trigger_runtime PASSED   [ 28%]
tests/test_runtime.py::test_missing_checkpoint_reports_logical_package_path PASSED [ 42%]
tests/test_runtime.py::test_missing_finetuned_checkpoint_reports_for_filesystem_resource PASSED [ 57%]
tests/test_runtime.py::test_missing_finetuned_checkpoint_reports_for_traversable_resource PASSED [ 71%]
tests/test_package_metadata.py::test_package_exposes_pep440_version PASSED [ 85%]
tests/test_package_metadata.py::test_plain_import_does_not_load_model_runtime PASSED [100%]

============================== 7 passed in 0.02s ===============================
```

Full-suite command:

```bash
python -m pytest -v
```

Full-suite output:

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.11, pytest-8.2.0, pluggy-1.6.0 -- /opt/homebrew/Caskroom/miniforge/base/bin/python
cachedir: .pytest_cache
rootdir: /Users/yueyang/projects/ai_agents/tsac/github/TeNNet-SAC
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.13.0
collecting ... collected 18 items

tests/test_package_metadata.py::test_package_exposes_pep440_version PASSED [  5%]
tests/test_package_metadata.py::test_plain_import_does_not_load_model_runtime PASSED [ 11%]
tests/test_runtime.py::test_get_runtime_builds_once PASSED               [ 16%]
tests/test_runtime.py::test_plain_import_cannot_trigger_runtime PASSED   [ 22%]
tests/test_runtime.py::test_missing_checkpoint_reports_logical_package_path PASSED [ 27%]
tests/test_runtime.py::test_missing_finetuned_checkpoint_reports_for_filesystem_resource PASSED [ 33%]
tests/test_runtime.py::test_missing_finetuned_checkpoint_reports_for_traversable_resource PASSED [ 38%]
tests/test_smi_ted_loading.py::test_load_smi_ted_prefers_requested_local_checkpoint PASSED [ 44%]
tests/test_smi_ted_loading.py::test_load_smi_ted_downloads_the_requested_checkpoint_name PASSED [ 50%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_preserves_legacy_smi_ted_tokens_and_ids[CCO-expected_tokens0-expected_ids0] PASSED [ 55%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_preserves_legacy_smi_ted_tokens_and_ids[ClCCCl-expected_tokens1-expected_ids1] PASSED [ 61%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_preserves_legacy_smi_ted_tokens_and_ids[C[C@H](O)C(=O)O-expected_tokens2-expected_ids2] PASSED [ 66%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_preserves_legacy_smi_ted_tokens_and_ids[[NH4+]-expected_tokens3-expected_ids3] PASSED [ 72%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_preserves_legacy_smi_ted_tokens_and_ids[C%12CCCCC%12-expected_tokens4-expected_ids4] PASSED [ 77%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_matches_legacy_batch_padding PASSED [ 83%]
tests/test_smi_ted_tokenizer.py::test_tokenizer_keeps_eos_when_truncating PASSED [ 88%]
tests/test_source_migration.py::test_only_src_tree_contains_runtime_sources PASSED [ 94%]
tests/test_source_migration.py::test_package_tree_contains_required_sources PASSED [100%]

============================== 18 passed in 1.29s ==============================
```

### Files changed

- `src/tennetsac/runtime.py`: added one canonical thirteen-resource list and validates every logical path before returning a checkpoint root; extraction reuses the same list.
- `tests/test_runtime.py`: added filesystem and non-filesystem missing fine-tuned resource coverage.
- `.superpowers/sdd/2026-09-06-unified-package-release/task-3-report.md`: appended this Fix Round 1 record.

### Self-review and concerns

- The validation and extraction resource lists cannot drift because `_checkpoint_path()` now reuses `_CHECKPOINT_RESOURCES` from `_checkpoint_root()`.
- Both tests assert the logical fine-tuned checkpoint name and package-relative checkpoint directory in the exception.
- `git diff --check` will be run before commit.
- As in the original implementation, full neural model construction and a real installed-wheel load are not exercised in this dependency/network-free test suite.
