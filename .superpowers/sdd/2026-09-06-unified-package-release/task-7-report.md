# Task 7 report: examples and repository documentation

## TDD evidence

- **RED:** Added `tests/test_documentation.py`, then ran
  `python -m pytest tests/test_documentation.py -v`. All three tests failed as
  intended: the README still linked the root notebook, the expected
  `examples/TeNNetSAC.ipynb` file did not exist, and `requirements.txt`
  repeated runtime dependencies instead of using the development extra.
- **GREEN:** Moved the notebook with `git mv`, changed it to import only
  `binary_lng`, `multi_lng`, and `profile` from `tennetsac`, and updated the
  README, requirements file, and environment. Re-running
  `python -m pytest tests/test_documentation.py -v` produced **3 passed**.
- **Full suite:** `python -m pytest -m 'not integration' -v` produced
  **108 passed, 2 deselected** in 3.03 seconds.

## Files changed

- `README.md`: documents PyPI and editable development installs, the moved
  notebook, lazy initialization, revision-pinned external models, the reserved
  empty offline ChemBERTa2 location, Git-tag versions, and the `src/tennetsac`
  layout.
- `examples/TeNNetSAC.ipynb`: moved from the repository root and normalized to
  the public API without source-tree imports or checkpoint paths. Existing
  prediction outputs were retained; the obsolete message claiming that imports
  had loaded every model was removed because it is stale under lazy loading.
- `requirements.txt`: now contains only `-e .[dev]`; runtime constraints remain
  owned by `pyproject.toml`.
- `TeNNet-SAC.yml`: retains Python 3.11 and the PyTorch/CUDA environment intent,
  but installs the local project via pip (`-e .[dev]`) instead of restating
  package dependencies.
- `tests/test_documentation.py`: regression coverage for the public import,
  notebook location/imports, and development requirements entry point.

## Encoding and format

`TeNNet-SAC.yml` was tracked as UTF-16LE with CRLF line endings. It is now
UTF-8 ASCII YAML with LF line endings, parsed successfully with `yaml.safe_load`.
It was staged explicitly with `git add -f` because the repository ignores
`*.yml` generally.

## Self-review

- Reviewed the staged name/status and diff summary.
- Ran `git diff --cached --check` with no whitespace errors.
- Validated the notebook with `python -m json.tool` and the environment with
  `yaml.safe_load`.
- Verified that the notebook has no `from core`, `from utils`, `from models`,
  `sys.path`, or checkpoint-path source references through the documentation
  regression test.

## Concerns

The existing Colab link remains unchanged; it may need a separate update if the
notebook is republished at a new Colab location. No notebook execution or model
download was performed.
