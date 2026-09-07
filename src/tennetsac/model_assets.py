"""Command-line and public torch-free API for external model assets."""

from __future__ import annotations

import argparse
import sys

from ._model_assets import (
    ModelAssetError,
    ResolvedModelAsset,
    asset_cache_path,
    resolve_model_asset,
    verify_model_asset,
)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        prog="python -m tennetsac.model_assets",
        description="Download or verify pinned TeNNet-SAC model assets.",
    )
    commands = argument_parser.add_subparsers(dest="command", required=True)
    for command in ("download", "verify"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument("model", choices=("smi-ted-light",))
    return argument_parser


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        resolved = (
            resolve_model_asset(args.model, allow_download=True)
            if args.command == "download"
            else verify_model_asset(args.model)
        )
    except ModelAssetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"path={resolved.path}")
    print(f"sha256={resolved.sha256}")
    return 0


__all__ = [
    "ModelAssetError",
    "ResolvedModelAsset",
    "asset_cache_path",
    "resolve_model_asset",
    "verify_model_asset",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
