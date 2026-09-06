import re

from tennetsac.model_manifest import (
    external_model,
    load_manifest,
    verify_bundled_artifacts,
)


EXPECTED_ARTIFACTS = {
    "gamma-base": ("ckpt_files/base.ckpt", "791ce5bf9c59e2099467882b4e7ee220575f464f2f3e007495a78ed9ae6f2d93"),
    "geometry": ("ckpt_files/geo.ckpt", "ab4e37731eb07573cf4d9a7a15c906c6ef8b0ae5a895ff4f921a736c32483624"),
    "sigma-profile": ("ckpt_files/prf.ckpt", "649e7139cc43a95bd459d0eb0c58fac3a5d3a62a9b15ae6bdb0018894c42723e"),
    "gamma-tuned-1": ("ckpt_files/fine-tuned/1.ckpt", "134be46cce6839a7b08250750e4803c8198a4725d5973ca124db76dd488aedc1"),
    "gamma-tuned-2": ("ckpt_files/fine-tuned/2.ckpt", "d763688bbe2898fc182ae6e9b39436a678ad892510c76b510c43c389e87d4c5b"),
    "gamma-tuned-3": ("ckpt_files/fine-tuned/3.ckpt", "15232df78dbbce274818e00bdfff455a163fbb9ff8af1ef3ba4de73650519400"),
    "gamma-tuned-4": ("ckpt_files/fine-tuned/4.ckpt", "bf01d2c6832b161ddf12d789dc886a3c3065a1810948d196f103a89c94f0daeb"),
    "gamma-tuned-5": ("ckpt_files/fine-tuned/5.ckpt", "937a02283521bba254b917685f4db265c48617732f9c5bbfffb14eaaf1d38813"),
    "gamma-tuned-6": ("ckpt_files/fine-tuned/6.ckpt", "7588bfe7a265752395373cf930ede2e721da2cb37fd7779a7b3b94d6c3590e79"),
    "gamma-tuned-7": ("ckpt_files/fine-tuned/7.ckpt", "f4c03f8281b7ac56213e5e04123c208fc4f2030ca610684bd1cbd9aa59f7e16d"),
    "gamma-tuned-8": ("ckpt_files/fine-tuned/8.ckpt", "0ff319a81281b4315638281283433590e6adf7b7ab17039919601747f736096d"),
    "gamma-tuned-9": ("ckpt_files/fine-tuned/9.ckpt", "73c65639135717a3eaa7130c03ea6f85fa3f95ed1b5937b5b73611bd33378c70"),
    "gamma-tuned-10": ("ckpt_files/fine-tuned/10.ckpt", "def4bb97274011564b33d55666bcc5d4d2bcae9c422cbf2c96529c7c364ebfcc"),
}


def test_manifest_describes_the_complete_bundled_checkpoint_set():
    manifest = load_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["bundle_version"] == "1.0.0"
    artifacts = {entry["name"]: (entry["path"], entry["sha256"]) for entry in manifest["artifacts"]}
    assert artifacts == EXPECTED_ARTIFACTS
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for _, digest in artifacts.values())
    assert verify_bundled_artifacts() == []


def test_manifest_pins_external_model_sources_and_revisions():
    assert external_model("chemberta2") == {
        "name": "chemberta2",
        "source": "DeepChem/ChemBERTa-77M-MLM",
        "revision": "ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
        "distribution": "external",
    }
    assert external_model("smi-ted-light") == {
        "name": "smi-ted-light",
        "source": "ibm/materials.smi-ted",
        "filename": "smi-ted-Light_40.pt",
        "revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
        "sha256": "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
        "distribution": "external",
    }
