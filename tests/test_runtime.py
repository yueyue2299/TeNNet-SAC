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


def test_missing_finetuned_checkpoint_reports_for_filesystem_resource(
    monkeypatch, tmp_path
):
    from tennetsac import runtime

    checkpoint_root = tmp_path / "ckpt_files"
    (checkpoint_root / "fine-tuned").mkdir(parents=True)
    for name in ("base.ckpt", "geo.ckpt", "prf.ckpt"):
        (checkpoint_root / name).touch()
    for index in range(1, 10):
        (checkpoint_root / "fine-tuned" / f"{index}.ckpt").touch()
    monkeypatch.setattr(runtime, "files", lambda package: tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match=r"fine-tuned/10\.ckpt.*tennetsac/ckpt_files",
    ):
        runtime._checkpoint_root()


def test_missing_finetuned_checkpoint_reports_for_traversable_resource(monkeypatch):
    from tennetsac import runtime

    class Resource:
        def __init__(self, path, available):
            self.path = path
            self.available = available

        def joinpath(self, name):
            path = "/".join(part for part in (self.path, name) if part)
            return Resource(path, self.available)

        def is_file(self):
            return self.path in self.available

    available = {
        "ckpt_files/base.ckpt",
        "ckpt_files/geo.ckpt",
        "ckpt_files/prf.ckpt",
        *(f"ckpt_files/fine-tuned/{index}.ckpt" for index in range(1, 10)),
    }
    monkeypatch.setattr(runtime, "files", lambda package: Resource("", available))

    with pytest.raises(
        FileNotFoundError,
        match=r"fine-tuned/10\.ckpt.*tennetsac/ckpt_files",
    ):
        runtime._checkpoint_root()
