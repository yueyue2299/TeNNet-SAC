from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory


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
    required = ("base.ckpt", "geo.ckpt", "prf.ckpt")
    missing = [name for name in required if not root.joinpath(name).is_file()]
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
        resources = (
            "base.ckpt",
            "geo.ckpt",
            "prf.ckpt",
            *(f"fine-tuned/{index}.ckpt" for index in range(1, 11)),
        )
        for resource_name in resources:
            resource = root
            for component in resource_name.split("/"):
                resource = resource.joinpath(component)
            destination = checkpoint_path / resource_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with as_file(resource) as source:
                copyfile(source, destination)
        yield checkpoint_path


def _build_runtime() -> Runtime:
    from .models.Emb2Geometry import GeometryGenerator
    from .models.Emb2Profile import SigmaProfileGenerator
    from .models.Prf2Gamma import Prf_to_Seg_Model
    from .utils.embedding import ChemBERTaEmbedder, SMITEDEmbedder
    from .utils.model_io import load_all_Gamma_models, load_model

    with _checkpoint_path(_checkpoint_root()) as checkpoint_root:
        return Runtime(
            chemberta_embedder=ChemBERTaEmbedder(),
            smi_ted_embedder=SMITEDEmbedder(),
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
