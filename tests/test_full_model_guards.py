import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace


INTEGRATION_TEST = Path(__file__).parent / "integration" / "test_full_model.py"


def _load_integration_module():
    spec = importlib.util.spec_from_file_location("full_model_guards", INTEGRATION_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_cache_guard_sets_offline_before_local_only_load(monkeypatch, tmp_path):
    checkpoint = tmp_path / "smi-ted-Light_40.pt"
    checkpoint.touch()
    monkeypatch.setenv("TENNETSAC_SMI_TED_CHECKPOINT", str(checkpoint))

    def assert_offline(*args, **kwargs):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    def assert_local_only(*args, **kwargs):
        assert_offline()
        assert kwargs.get("local_files_only") is True
        return object()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            try_to_load_from_cache=lambda *args, **kwargs: assert_offline()
            or "/cache/config.json"
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            RobertaTokenizer=SimpleNamespace(from_pretrained=assert_local_only),
            RobertaModel=SimpleNamespace(from_pretrained=assert_local_only),
        ),
    )

    _load_integration_module()._require_local_models(monkeypatch)
