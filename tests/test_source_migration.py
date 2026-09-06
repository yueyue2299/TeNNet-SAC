from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_only_src_tree_contains_runtime_sources():
    for old_path in ("models", "utils", "smi_ted_light", "ckpt_files"):
        assert not (ROOT / old_path).exists()


def test_package_tree_contains_required_sources():
    package = ROOT / "src" / "tennetsac"
    expected = {
        "core.py",
        "models/Emb2Geometry.py",
        "models/Emb2Profile.py",
        "models/Prf2Gamma.py",
        "utils/embedding.py",
        "utils/model_io.py",
        "utils/property.py",
        "utils/smiles.py",
        "smi_ted_light/load.py",
        "smi_ted_light/tokenizer.py",
        "smi_ted_light/bert_vocab_curated.txt",
        "ckpt_files/base.ckpt",
        "ckpt_files/geo.ckpt",
        "ckpt_files/prf.ckpt",
    }
    assert not [path for path in expected if not (package / path).is_file()]
