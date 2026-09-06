from pathlib import Path

import yaml


WORKFLOW_PATH = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"


def _workflow():
    with WORKFLOW_PATH.open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.BaseLoader)


def _run_commands(job):
    return [
        step["run"]
        for step in job["steps"]
        if isinstance(step, dict) and "run" in step
    ]


def test_ci_triggers_and_permissions_are_reusable():
    workflow = _workflow()

    assert set(workflow["on"]) == {"pull_request", "push", "workflow_call"}
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["env"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONWARNINGS": "error::ResourceWarning",
    }


def test_unit_job_covers_supported_python_versions_and_skips_integration():
    workflow = _workflow()
    unit = workflow["jobs"]["unit"]

    assert unit["strategy"]["matrix"]["python-version"] == [
        "3.10",
        "3.11",
        "3.12",
    ]
    assert any(
        'python -m pytest -m "not integration" -v' in command
        for command in _run_commands(unit)
    )


def test_distribution_job_builds_checks_and_smoke_tests_installed_wheel():
    workflow = _workflow()
    distribution = workflow["jobs"]["distribution"]
    commands = _run_commands(distribution)
    command_text = "\n".join(commands)

    assert "python -m build" in command_text
    assert "python -m twine check dist/*" in command_text
    assert "python scripts/verify_distribution.py dist/*" in command_text
    assert "pip install" in command_text and "*.whl" in command_text
    assert "mktemp" in command_text
    assert "cd \"$smoke_dir\"" in command_text or "cd \"${smoke_dir}\"" in command_text
    assert "import tennetsac" in command_text


def test_tokenizer_compatibility_isolated_matrix_and_targeted_test():
    workflow = _workflow()
    tokenizer = workflow["jobs"]["tokenizer-compatibility"]
    versions = tokenizer["strategy"]["matrix"]["transformers"]
    commands = _run_commands(tokenizer)
    command_text = "\n".join(commands)

    assert versions == ["transformers==4.36.2", "transformers>=5,<6"]
    assert "pip install --no-deps ." in command_text
    assert "tests/test_smi_ted_tokenizer.py" in command_text
    assert "python -m pytest tests/test_smi_ted_tokenizer.py -v" in command_text
