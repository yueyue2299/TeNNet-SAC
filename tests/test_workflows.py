from pathlib import Path

import yaml


WORKFLOW_PATH = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"
WORKFLOW_DIR = WORKFLOW_PATH.parent


def _workflow(path=WORKFLOW_PATH):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.BaseLoader)


def _named_workflow(name):
    path = WORKFLOW_DIR / name
    if not path.exists():
        raise AssertionError(f"missing workflow: {name}")
    return _workflow(path)


def _run_commands(job):
    return [
        step["run"]
        for step in job["steps"]
        if isinstance(step, dict) and "run" in step
    ]


def _uses_steps(job):
    return [
        step
        for step in job["steps"]
        if isinstance(step, dict) and "uses" in step
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


def test_release_build_triggers_only_for_version_tags_and_cannot_publish():
    workflow = _named_workflow("release-build.yml")

    assert workflow["on"] == {"push": {"tags": ["v*"]}}
    assert workflow["permissions"] == {"contents": "write"}
    assert "id-token" not in workflow["permissions"]


def test_release_build_reuses_ci_and_constructs_draft_release_once():
    workflow = _named_workflow("release-build.yml")
    ci_job = workflow["jobs"]["ci"]
    release = workflow["jobs"]["release"]
    commands = _run_commands(release)
    command_text = "\n".join(commands)

    assert ci_job["uses"] == "./.github/workflows/ci.yml"
    assert release["needs"] == ["ci"]
    assert command_text.count("python -m build") == 1
    assert 'python -m pytest -m "not integration" -v' in commands
    assert "python -m twine check dist/*" in commands
    assert "python scripts/verify_distribution.py dist/*" in commands
    assert (
        "python -c 'from tennetsac.model_manifest import verify_bundled_artifacts; "
        "errors = verify_bundled_artifacts(); assert not errors, errors'"
        in commands
    )
    assert 'python scripts/verify_release_version.py --tag "$GITHUB_REF_NAME" dist/*' in commands
    assert "python -m pip install --force-reinstall --no-deps dist/*.whl" in commands
    assert (
        "python -c 'import os, tennetsac; assert tennetsac.__version__ == "
        'os.environ["GITHUB_REF_NAME"].removeprefix("v")\''
        in commands
    )
    assert "cd dist && shasum -a 256 * > SHA256SUMS" in commands
    assert (
        'gh release create "$GITHUB_REF_NAME" dist/*.whl dist/*.tar.gz '
        "dist/SHA256SUMS --draft --verify-tag"
        in commands
    )
    assert "python -m build" not in "\n".join(
        step.get("run", "") for step in _uses_steps(release)
    )


def test_publish_pypi_is_manual_protected_trusted_publishing():
    workflow = _named_workflow("publish-pypi.yml")
    job = workflow["jobs"]["publish"]

    assert workflow["on"] == {
        "workflow_dispatch": {
            "inputs": {
                "tag": {
                    "description": "Existing GitHub Release tag to publish",
                    "required": "true",
                    "type": "string",
                }
            }
        }
    }
    assert workflow["permissions"] == {"contents": "read", "id-token": "write"}
    assert job["environment"] == "pypi"


def test_publish_pypi_downloads_verifies_and_publishes_existing_assets_only():
    workflow = _named_workflow("publish-pypi.yml")
    publish = workflow["jobs"]["publish"]
    commands = _run_commands(publish)
    command_text = "\n".join(commands)
    uses = [step["uses"] for step in _uses_steps(publish)]
    tag_validation = (
        "python -c 'import os; from scripts.verify_release_version import version_from_tag; "
        'version_from_tag(os.environ["RELEASE_TAG"])\''
    )

    assert tag_validation in commands
    assert commands.index(tag_validation) < commands.index(
        'gh release download "$RELEASE_TAG" --dir dist'
    )
    assert 'gh release download "$RELEASE_TAG" --dir dist' in commands
    assert (
        "python -c 'from pathlib import Path; extras = [p.name for p in Path(\"dist\").iterdir() "
        "if p.name != \"SHA256SUMS\" and p.suffix != \".whl\" and not p.name.endswith(\".tar.gz\")]; "
        "assert not extras, extras'"
        in commands
    )
    assert "cd dist && shasum -a 256 -c SHA256SUMS" in commands
    assert 'python scripts/verify_release_version.py --tag "$RELEASE_TAG" dist/*.whl dist/*.tar.gz' in commands
    assert "python -m twine check dist/*.whl dist/*.tar.gz" in commands
    assert "rm dist/SHA256SUMS" in commands
    assert "pypa/gh-action-pypi-publish@release/v1" in uses
    publisher = next(
        step for step in _uses_steps(publish)
        if step["uses"] == "pypa/gh-action-pypi-publish@release/v1"
    )
    assert publisher["with"] == {"packages-dir": "dist/"}
    forbidden_build_tokens = ("python -m build", " sdist", " bdist")
    assert not any(token in command_text for token in forbidden_build_tokens)
    assert not any(command.strip() == "build" for command in commands)
