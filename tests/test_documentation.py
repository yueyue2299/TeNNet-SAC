"""Regression checks for the documented, installable-package workflow."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_the_public_package_and_notebook_location() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "from tennetsac import" in readme
    assert "examples/TeNNetSAC.ipynb" in readme
    assert "`src/tennetsac`" in readme


def test_readme_documents_the_smi_ted_asset_workflow_and_immutable_updates() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_facts = {
        "python -m tennetsac.model_assets download smi-ted-light",
        "python -m tennetsac.model_assets verify smi-ted-light",
        "TENNETSAC_CACHE_DIR",
        "TENNETSAC_SMI_TED_CHECKPOINT",
        "TENNETSAC_OFFLINE=1",
        "model-smi-ted-light-v1",
        "first prediction call",
        "next major package version",
        "model-smi-ted-light-v2",
        "package patch",
        "never moved or replaced",
    }

    assert not sorted(fact for fact in required_facts if fact not in readme)


def _notebook() -> dict:
    return json.loads(
        (ROOT / "examples" / "TeNNetSAC.ipynb").read_text(encoding="utf-8")
    )


def test_example_notebook_uses_only_the_public_package_api() -> None:
    notebook = _notebook()
    notebook_source = "".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )

    assert "from tennetsac import binary_lng, multi_lng, profile" in notebook_source
    assert "from core import" not in notebook_source
    assert "from utils" not in notebook_source
    assert "from models" not in notebook_source
    assert "sys.path" not in notebook_source
    assert "ckpt_path" not in notebook_source


def test_example_notebook_uses_the_keyword_multicomponent_contract() -> None:
    notebook = _notebook()
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(code)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "multi_lng"
    ]

    assert len(calls) == 1
    call = calls[0]
    assert call.args == []
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords} == {
        "smiles": "smiles_list",
        "temperature": "temperature",
        "composition": "mole_fraction_list",
        "version": "version",
    }
    assert "model_type" not in notebook_source


def test_example_notebook_contains_no_retained_execution_state() -> None:
    code_cells = [
        cell for cell in _notebook()["cells"] if cell["cell_type"] == "code"
    ]

    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs") == [] for cell in code_cells)


def test_multi_lng_docstring_documents_both_composition_lengths() -> None:
    tree = ast.parse(
        (ROOT / "src" / "tennetsac" / "core.py").read_text(encoding="utf-8")
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "multi_lng"
    )
    docstring = ast.get_docstring(function)

    assert docstring is not None
    assert "N fractions" in docstring
    assert "N-1 fractions" in docstring
    assert "1 - sum(composition)" in docstring


def test_requirements_installs_the_package_with_development_extras() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()

    assert requirements == ["-e .[dev]"]
