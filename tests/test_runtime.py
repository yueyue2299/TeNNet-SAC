import importlib

import pytest


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
    import tennetsac.runtime as runtime

    runtime.get_runtime.cache_clear()
    monkeypatch.setattr(
        runtime,
        "_build_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("runtime initialized")),
    )
    importlib.reload(importlib.import_module("tennetsac"))


def test_missing_checkpoint_reports_logical_package_path(monkeypatch):
    from tennetsac import runtime

    class MissingRoot:
        def joinpath(self, name):
            return self

        def is_file(self):
            return False

    monkeypatch.setattr(runtime, "files", lambda package: MissingRoot())

    with pytest.raises(FileNotFoundError, match=r"base\.ckpt.*tennetsac/ckpt_files"):
        runtime._checkpoint_root()
