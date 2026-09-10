from pathlib import Path
import re
import subprocess

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
        for step in job.get("steps", [])
        if isinstance(step, dict) and "uses" in step
    ]


def _step(job, name):
    return next(step for step in job["steps"] if step.get("name") == name)


def _command_lines(command):
    return [line.strip() for line in command.splitlines() if line.strip()]


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

    assert _step(distribution, "Build wheel and source distribution")["run"] == (
        "python -m build"
    )
    assert _step(distribution, "Check distribution metadata")["run"] == (
        "python -m twine check dist/*"
    )
    assert _step(distribution, "Verify distribution contents")["run"] == (
        "python scripts/verify_distribution.py dist/*"
    )

    smoke = _command_lines(
        _step(distribution, "Verify the installed wheel outside the checkout")["run"]
    )
    assert smoke[:4] == [
        'smoke_dir="$(mktemp -d)"',
        'python -m pip install --no-deps --target "$smoke_dir" dist/*.whl',
        "(",
        'cd "$smoke_dir"',
    ]
    assert 'PYTHONPATH="$smoke_dir" python - <<\'PY\'' in smoke
    assert "from tennetsac.model_manifest import verify_bundled_artifacts" in smoke
    assert "errors = verify_bundled_artifacts()" in smoke
    assert any("raise SystemExit" in line for line in smoke)
    assert not any(re.search(r"\bassert\b", line) for line in smoke)


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
    release = workflow["jobs"]["release"]

    assert workflow["on"] == {"push": {"tags": ["v*"]}}
    assert workflow["permissions"] == {"contents": "read"}
    assert "id-token" not in workflow["permissions"]
    assert release["permissions"] == {"contents": "write"}


def test_release_build_reuses_ci_and_constructs_draft_release_once():
    workflow = _named_workflow("release-build.yml")
    ci_job = workflow["jobs"]["ci"]
    release = workflow["jobs"]["release"]
    commands = _run_commands(release)
    command_text = "\n".join(commands)
    upload = next(
        step for step in _uses_steps(release)
        if step["uses"].startswith("actions/upload-artifact@")
    )

    assert ci_job["uses"] == "./.github/workflows/ci.yml"
    assert release["needs"] == ["ci"]
    assert command_text.count("python -m build") == 1
    assert 'python -m pytest -m "not integration" -v' in commands
    assert "python -m twine check dist/*" in commands
    assert "python scripts/verify_distribution.py dist/*" in commands
    assert 'python scripts/verify_release_version.py --tag "$GITHUB_REF_NAME" dist/*' in commands
    installed = _command_lines(
        _step(release, "Verify the installed release wheel")["run"]
    )
    assert installed[:4] == [
        'release_dir="$(mktemp -d)"',
        'python -m pip install --no-deps --target "$release_dir" dist/*.whl',
        "(",
        'cd "$release_dir"',
    ]
    assert (
        'PYTHONPATH="$release_dir" GITHUB_REF_NAME="$GITHUB_REF_NAME" '
        "python - <<'PY'"
    ) in installed
    assert "from tennetsac.model_manifest import verify_bundled_artifacts" in installed
    assert "errors = verify_bundled_artifacts()" in installed
    assert any("tennetsac.__version__" in line for line in installed)
    assert any("raise SystemExit" in line for line in installed)
    assert not any(re.search(r"\bassert\b", line) for line in installed)
    assert "cd dist && shasum -a 256 * > SHA256SUMS" in commands
    assert upload["with"] == {
        "name": "tennetsac-release-${{ github.ref_name }}",
        "path": "dist/*.whl\ndist/*.tar.gz\ndist/SHA256SUMS\n",
        "if-no-files-found": "error",
    }
    assert (
        'gh release create "$GITHUB_REF_NAME" dist/*.whl dist/*.tar.gz '
        "dist/SHA256SUMS --draft --verify-tag"
        in commands
    )
    assert "python -m build" not in "\n".join(
        step.get("run", "") for step in _uses_steps(release)
    )


def test_release_build_verifies_public_smi_ted_asset_before_packaging():
    workflow = _named_workflow("release-build.yml")
    release = workflow["jobs"]["release"]
    commands = _run_commands(release)
    unit_tests = 'python -m pytest -m "not integration" -v'
    build = "python -m build"
    draft_release = (
        'gh release create "$GITHUB_REF_NAME" dist/*.whl dist/*.tar.gz '
        "dist/SHA256SUMS --draft --verify-tag"
    )
    smoke = _step(release, "Verify public SMI-TED asset before packaging")

    assert release.get("env") is None
    assert smoke["env"] == {
        "TENNETSAC_OFFLINE": "0",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "TENNETSAC_PUBLIC_ASSET_SMOKE": "1",
        "TENNETSAC_SMI_TED_CHECKPOINT": "",
        "TENNETSAC_CACHE_DIR": "${{ runner.temp }}/tennetsac-public-asset-smoke",
    }
    assert _command_lines(smoke["run"]) == [
        "python -m tennetsac.model_assets download smi-ted-light",
        "python -m tennetsac.model_assets verify smi-ted-light",
        "python -m pytest tests/integration/test_smi_ted_asset.py -k public_asset -v",
        "python -m pytest tests/integration/test_full_model.py -k public_asset -v",
    ]
    assert commands.index(unit_tests) < commands.index(smoke["run"])
    assert commands.index(smoke["run"]) < commands.index(build)
    assert commands.index(smoke["run"]) < commands.index(draft_release)

    for step in release["steps"]:
        if step is smoke:
            continue
        assert not set(step.get("env", {})).intersection(smoke["env"])


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
    tag_validation = (
        "python -c 'import os; from scripts.verify_release_version import version_from_tag; "
        'version_from_tag(os.environ["RELEASE_TAG"])\''
    )
    checksum_manifest_validation = (
        "python scripts/verify_release_version.py --verify-sha256sums "
        "dist/SHA256SUMS dist/*.whl dist/*.tar.gz"
    )
    unexpected_asset_validation = next(
        command for command in commands
        if "unexpected release assets" in command
    )

    assert tag_validation in commands
    assert commands.index(tag_validation) < commands.index(
        'gh release download "$RELEASE_TAG" --dir dist'
    )
    assert 'gh release download "$RELEASE_TAG" --dir dist' in commands
    assert "raise SystemExit" in unexpected_asset_validation
    assert "assert not extras" not in unexpected_asset_validation
    assert checksum_manifest_validation in commands
    assert commands.index(checksum_manifest_validation) < commands.index(
        "cd dist && shasum -a 256 -c SHA256SUMS"
    )
    assert "cd dist && shasum -a 256 -c SHA256SUMS" in commands
    assert 'python scripts/verify_release_version.py --tag "$RELEASE_TAG" dist/*.whl dist/*.tar.gz' in commands
    assert "python -m twine check dist/*.whl dist/*.tar.gz" in commands
    distribution_validation = (
        "python scripts/verify_distribution.py dist/*.whl dist/*.tar.gz"
    )
    assert distribution_validation in commands
    assert commands.index(distribution_validation) < commands.index(
        "python -m twine check dist/*.whl dist/*.tar.gz"
    )
    assert "rm dist/SHA256SUMS" in commands
    publisher = next(
        step for step in _uses_steps(publish)
        if step["uses"].startswith("pypa/gh-action-pypi-publish@")
    )
    assert publisher["with"] == {"packages-dir": "dist/"}
    forbidden_build_tokens = ("python -m build", " sdist", " bdist")
    assert not any(token in command_text for token in forbidden_build_tokens)
    assert not any(command.strip() == "build" for command in commands)


def test_external_actions_are_immutable_and_audited() -> None:
    expected = {
        "actions/checkout": (
            "3d3c42e5aac5ba805825da76410c181273ba90b1",
            "v7.0.1",
        ),
        "actions/setup-python": (
            "5fda3b95a4ea91299a34e894583c3862153e4b97",
            "v7.0.0",
        ),
        "actions/upload-artifact": (
            "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
            "v7.0.1",
        ),
        "pypa/gh-action-pypi-publish": (
            "dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
            "v1.14.2",
        ),
    }
    seen = set()

    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        source = path.read_text(encoding="utf-8")
        workflow = _workflow(path)
        for job in workflow["jobs"].values():
            references = [job["uses"]] if "uses" in job else []
            references.extend(step["uses"] for step in _uses_steps(job))
            for reference in references:
                if reference.startswith("./"):
                    continue
                action, separator, commit = reference.partition("@")
                assert separator and re.fullmatch(r"[0-9a-f]{40}", commit)
                assert action in expected
                expected_commit, version = expected[action]
                assert commit == expected_commit
                assert f"uses: {reference} # {version}" in source
                seen.add(action)

    assert seen == set(expected)


def test_dependabot_checks_for_github_actions_updates_weekly() -> None:
    path = WORKFLOW_DIR.parent / "dependabot.yml"
    with path.open(encoding="utf-8") as stream:
        configuration = yaml.load(stream, Loader=yaml.BaseLoader)

    assert configuration["version"] == "2"
    assert configuration["updates"] == [
        {
            "package-ecosystem": "github-actions",
            "directory": "/",
            "schedule": {"interval": "weekly"},
        }
    ]


def test_dependabot_configuration_is_not_gitignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", ".github/dependabot.yml"],
        cwd=WORKFLOW_DIR.parents[1],
        check=False,
    )

    assert result.returncode == 1


def test_release_critical_workflow_python_does_not_use_assert() -> None:
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        workflow = _workflow(path)
        for job in workflow["jobs"].values():
            if "steps" not in job:
                continue
            for command in _run_commands(job):
                assert not re.search(r"\bassert\b", command)
