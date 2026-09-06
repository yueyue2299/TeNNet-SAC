import sys
import subprocess

from packaging.version import Version


def test_package_exposes_pep440_version():
    import tennetsac

    assert Version(tennetsac.__version__)


def test_plain_import_does_not_load_model_runtime():
    code = """
import sys
import tennetsac
assert "tennetsac.runtime" not in sys.modules
assert "tennetsac.core" not in sys.modules
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
assert set(tennetsac.__all__) == {
    "profile",
    "binary_lng",
    "multi_lng",
    "fit_nrtl",
    "plot_nrtl_fitting",
    "__version__",
}
"""
    subprocess.run([sys.executable, "-c", code], check=True)
