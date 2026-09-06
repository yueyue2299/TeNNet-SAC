from importlib import import_module

from ._version import __version__

__all__ = [
    "profile",
    "binary_lng",
    "multi_lng",
    "fit_nrtl",
    "plot_nrtl_fitting",
    "__version__",
]

_PUBLIC_FUNCTIONS = frozenset(__all__) - {"__version__"}


def __getattr__(name: str):
    if name not in _PUBLIC_FUNCTIONS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".core", __name__), name)
    globals()[name] = value
    return value
