from contextlib import contextmanager
import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def isolated_runtime_build(monkeypatch, tmp_path):
    from tennetsac import model_manifest, runtime
    from tennetsac.models import Emb2Geometry, Emb2Profile, Prf2Gamma
    from tennetsac.utils import embedding, model_io

    checkpoint_root = tmp_path / "ckpt_files"
    vocab_dir = tmp_path / "smi_ted_light"
    vocab_dir.mkdir()
    vocab_path = vocab_dir / "bert_vocab_curated.txt"
    vocab_path.write_text("<bos>\n<eos>\n<pad>\n<mask>\nC\n", encoding="utf-8")

    @contextmanager
    def checkpoint_path(_root):
        yield checkpoint_root

    @contextmanager
    def smi_ted_vocab_dir():
        yield vocab_dir

    monkeypatch.setattr(runtime, "_checkpoint_root", lambda: checkpoint_root)
    monkeypatch.setattr(runtime, "_checkpoint_path", checkpoint_path)
    monkeypatch.setattr(runtime, "_smi_ted_vocab_dir", smi_ted_vocab_dir)
    monkeypatch.setattr(Emb2Geometry, "GeometryGenerator", lambda: "geometry")
    monkeypatch.setattr(Emb2Profile, "SigmaProfileGenerator", lambda: "profile")
    monkeypatch.setattr(Prf2Gamma, "Prf_to_Seg_Model", lambda: "gamma")
    monkeypatch.setattr(
        model_io, "load_model", lambda model, path: (model, path.name)
    )
    monkeypatch.setattr(
        model_io,
        "load_all_Gamma_models",
        lambda model_class, path: ["fine-tuned"],
    )
    monkeypatch.setattr(
        embedding,
        "ChemBERTaEmbedder",
        lambda **kwargs: ("chemberta", kwargs),
    )
    monkeypatch.setattr(
        model_manifest,
        "external_model",
        lambda name: {
            "name": "chemberta2",
            "source": "DeepChem/ChemBERTa-77M-MLM",
            "revision": "pinned-revision",
        }
        if name == "chemberta2"
        else (_ for _ in ()).throw(
            AssertionError(f"runtime directly loaded unexpected manifest entry: {name}")
        ),
    )
    return SimpleNamespace(
        runtime=runtime,
        embedding=embedding,
        checkpoint_root=checkpoint_root,
        vocab_path=vocab_path,
    )


def test_get_runtime_builds_once(monkeypatch):
    from tennetsac import runtime

    marker = object()
    calls = []
    runtime.get_runtime.cache_clear()
    monkeypatch.setattr(runtime, "_build_runtime", lambda: calls.append(1) or marker)

    assert runtime.get_runtime() is marker
    assert runtime.get_runtime() is marker
    assert calls == [1]


def test_build_runtime_resolves_smi_ted_once_and_propagates_explicit_asset(
    monkeypatch, tmp_path, isolated_runtime_build
):
    from tennetsac import _model_assets

    build = isolated_runtime_build
    checkpoint_path = tmp_path / "explicit-parent.pt"
    asset_entry = {
        "name": "smi-ted-light",
        "parent": {"sha256": "pinned-parent-digest"},
    }
    resolved = SimpleNamespace(
        path=checkpoint_path,
        manifest_entry=asset_entry,
    )
    resolve_calls = []
    embedder_calls = []

    monkeypatch.setattr(
        _model_assets,
        "resolve_model_asset",
        lambda name: resolve_calls.append(name) or resolved,
    )
    monkeypatch.setattr(
        build.embedding,
        "SMITEDEmbedder",
        lambda **kwargs: embedder_calls.append(kwargs) or ("smi-ted", kwargs),
    )

    result = build.runtime._build_runtime()

    assert resolve_calls == ["smi-ted-light"]
    assert embedder_calls == [
        {
            "checkpoint_path": checkpoint_path,
            "vocab_path": build.vocab_path,
            "asset_entry": asset_entry,
        }
    ]
    assert result.smi_ted_embedder == ("smi-ted", embedder_calls[0])


def test_build_runtime_wraps_offline_smi_ted_failure_with_original_cause(
    monkeypatch, isolated_runtime_build
):
    from tennetsac import _model_assets

    build = isolated_runtime_build
    asset_error = _model_assets.ModelAssetError(
        "offline cache miss; run python -m tennetsac.model_assets download smi-ted-light"
    )
    resolve_calls = []

    def fail_resolution(name):
        resolve_calls.append(name)
        raise asset_error

    monkeypatch.setattr(_model_assets, "resolve_model_asset", fail_resolution)
    monkeypatch.setattr(
        build.embedding,
        "SMITEDEmbedder",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"embedder constructed after failed resolution: {kwargs}")
        ),
    )

    with pytest.raises(
        RuntimeError, match="Failed to initialize SMI-TED embedder"
    ) as error:
        build.runtime._build_runtime()

    assert resolve_calls == ["smi-ted-light"]
    assert error.value.__cause__ is asset_error
    assert "offline cache miss" in str(error.value.__cause__)
    assert "tennetsac.model_assets download smi-ted-light" in str(
        error.value.__cause__
    )


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
