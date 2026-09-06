"""Regression checks for the documented, installable-package workflow."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_the_public_package_and_notebook_location() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "from tennetsac import" in readme
    assert "examples/TeNNetSAC.ipynb" in readme
    assert "`src/tennetsac`" in readme


def test_example_notebook_uses_only_the_public_package_api() -> None:
    notebook = json.loads((ROOT / "examples" / "TeNNetSAC.ipynb").read_text(encoding="utf-8"))
    notebook_source = "".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )

    assert "from tennetsac import binary_lng, multi_lng, profile" in notebook_source
    assert "from core import" not in notebook_source
    assert "from utils" not in notebook_source
    assert "from models" not in notebook_source
    assert "sys.path" not in notebook_source
    assert "ckpt_path" not in notebook_source


def test_requirements_installs_the_package_with_development_extras() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()

    assert requirements == ["-e .[dev]"]
