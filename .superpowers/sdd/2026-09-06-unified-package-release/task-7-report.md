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

## Fix Round 1: N-1 multicomponent composition

The reviewer finding does not require a production or documentation change.
`multi_lng` forwards its `composition` to the real
`utils.property.calc_ln_gamma` path. That function already accepts exactly
`num_components - 1` fractions (lines 68--73), appends
`1.0 - sum(mole_fraction_list)`, and then performs the calculation. The PyPI
0.1.10 golden integration fixture likewise records the valid three-component
call with `[0.3, 0.4]`.

Added `test_multi_lng_autocompletes_an_n_minus_one_composition` to
`tests/test_api_contract.py`. It calls the public `tennetsac.multi_lng` API
with lightweight fake profiles and a fake predictor, while retaining the real
`calc_ln_gamma` calculation. It asserts that the N-1 call returns the same
three values as the explicit `[0.3, 0.4, 0.3]` call. No RED production failure
was available to demonstrate because the established implementation was already
correct; removing the N-1 branch would make the first call raise the existing
length-mismatch `ValueError`, so this regression test would fail.

Commands and results:

- `python -m pytest tests/test_api_contract.py::test_multi_lng_autocompletes_an_n_minus_one_composition -v`: **1 passed**.
- `python -m pytest tests/test_documentation.py tests/test_api_contract.py -v`: **10 passed**.
- `python -m pytest -m 'not integration' -v`: **109 passed, 2 deselected**.

Concern: the published README example is intentionally left at N-1 composition
to preserve the PyPI 0.1.10-compatible public behavior.
