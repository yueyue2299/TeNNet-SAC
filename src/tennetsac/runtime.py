from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
import os
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory


_CHECKPOINT_RESOURCES = (
    "base.ckpt",
    "geo.ckpt",
    "prf.ckpt",
    *(f"fine-tuned/{index}.ckpt" for index in range(1, 11)),
)


@dataclass(frozen=True)
class Runtime:
    chemberta_embedder: object
    smi_ted_embedder: object
    profile_model: object
    geometry_model: object
    gamma_base_model: object
    gamma_finetuned_models: tuple[object, ...]


def _checkpoint_root():
    root = files("tennetsac").joinpath("ckpt_files")
    missing = []
    for resource_name in _CHECKPOINT_RESOURCES:
        resource = root
        for component in resource_name.split("/"):
            resource = resource.joinpath(component)
        if not resource.is_file():
            missing.append(resource_name)
    if missing:
        raise FileNotFoundError(
            f"Missing packaged checkpoint(s) {missing} under tennetsac/ckpt_files"
        )
    return root


@contextmanager
def _checkpoint_path(root):
    """Yield a filesystem path for checkpoint resources on all supported Pythons."""
    if isinstance(root, Path):
        yield root
        return

    # Python 3.10 and 3.11 cannot pass a directory Traversable to as_file.
    # Materializing each file also works for zip-imported wheels on 3.12.
    with TemporaryDirectory() as temporary_dir:
        checkpoint_path = Path(temporary_dir)
        for resource_name in _CHECKPOINT_RESOURCES:
            resource = root
            for component in resource_name.split("/"):
                resource = resource.joinpath(component)
            destination = checkpoint_path / resource_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with as_file(resource) as source:
                copyfile(source, destination)
        yield checkpoint_path


@contextmanager
def _smi_ted_vocab_dir():
    vocab = files("tennetsac").joinpath(
        "smi_ted_light", "bert_vocab_curated.txt"
    )
    if not vocab.is_file():
        raise FileNotFoundError(
            "Missing packaged SMI-TED vocabulary under "
            "tennetsac/smi_ted_light/bert_vocab_curated.txt"
        )
    with as_file(vocab) as path:
        yield path.parent


def _build_runtime() -> Runtime:
    from .models.Emb2Geometry import GeometryGenerator
    from .models.Emb2Profile import SigmaProfileGenerator
    from .models.Prf2Gamma import Prf_to_Seg_Model
    from .utils.embedding import ChemBERTaEmbedder, SMITEDEmbedder
    from .utils.model_io import load_all_Gamma_models, load_model
    from .model_manifest import external_model

    with _checkpoint_path(_checkpoint_root()) as checkpoint_root:
        chemberta = external_model("chemberta2")
        smi_ted = external_model("smi-ted-light")
        try:
            chemberta_embedder = ChemBERTaEmbedder(
                model_name=chemberta["source"], revision=chemberta["revision"]
            )
        except Exception as error:
            raise RuntimeError("Failed to initialize ChemBERTa2 embedder") from error
        try:
            checkpoint_override = os.environ.get("TENNETSAC_SMI_TED_CHECKPOINT")
            if checkpoint_override:
                checkpoint_path = Path(checkpoint_override)
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        "TENNETSAC_SMI_TED_CHECKPOINT must name a file: "
                        f"{checkpoint_path}"
                    )
                with _smi_ted_vocab_dir() as vocab_dir:
                    smi_ted_embedder = SMITEDEmbedder(
                        model_dir=checkpoint_path.parent,
                        repo_id=smi_ted["source"],
                        revision=smi_ted["revision"],
                        ckpt_name=checkpoint_path.name,
                        expected_sha256=smi_ted["sha256"],
                        vocab_filename=vocab_dir / "bert_vocab_curated.txt",
                    )
            else:
                with _smi_ted_vocab_dir() as vocab_dir:
                    smi_ted_embedder = SMITEDEmbedder(
                        model_dir=vocab_dir,
                        repo_id=smi_ted["source"],
                        revision=smi_ted["revision"],
                        ckpt_name=smi_ted["filename"],
                        expected_sha256=smi_ted["sha256"],
                    )
        except Exception as error:
            raise RuntimeError("Failed to initialize SMI-TED embedder") from error
        return Runtime(
            chemberta_embedder=chemberta_embedder,
            smi_ted_embedder=smi_ted_embedder,
            profile_model=load_model(
                SigmaProfileGenerator(), checkpoint_root / "prf.ckpt"
            ),
            geometry_model=load_model(
                GeometryGenerator(), checkpoint_root / "geo.ckpt"
            ),
            gamma_base_model=load_model(
                Prf_to_Seg_Model(), checkpoint_root / "base.ckpt"
            ),
            gamma_finetuned_models=tuple(
                load_all_Gamma_models(
                    Prf_to_Seg_Model, checkpoint_root / "fine-tuned"
                )
            ),
        )


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return _build_runtime()
