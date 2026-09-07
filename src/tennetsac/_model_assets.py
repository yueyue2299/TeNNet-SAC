"""Torch-free resolution and verification for external model assets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen
from uuid import uuid4

from filelock import FileLock, Timeout
from platformdirs import user_cache_path

from .model_manifest import external_model


_DOWNLOAD_COMMAND = "python -m tennetsac.model_assets download {name}"
_LOCK_TIMEOUT_SECONDS = 120
_STREAM_CHUNK_BYTES = 1024 * 1024
_VERIFIED_HASHES: dict[tuple[str, int, int, str], tuple[int, int, int]] = {}


class ModelAssetError(RuntimeError):
    """Raised when an external model asset cannot be safely resolved."""


@dataclass(frozen=True)
class ResolvedModelAsset:
    path: Path
    sha256: str
    format: str
    is_legacy: bool
    manifest_entry: dict[str, Any]


def _clear_verified_hash_cache() -> None:
    """Clear per-process verification state (intended for isolated tests)."""

    _VERIFIED_HASHES.clear()


def asset_cache_path(asset_entry: dict[str, Any]) -> Path:
    configured = os.environ.get("TENNETSAC_CACHE_DIR")
    root = Path(configured).expanduser() if configured else user_cache_path("tennetsac")
    return (
        root
        / "models"
        / asset_entry["name"]
        / f"v{asset_entry['format_version']}"
        / asset_entry["filename"]
    )


def _error_message(
    entry: dict[str, Any],
    reason: str,
    *,
    location: str | Path,
    expected: str | None = None,
) -> str:
    expected_digest = expected or entry["sha256"]
    return (
        f"Model asset {entry['name']!r} ({entry['release_tag']}) {reason}. "
        f"URL/path: {location}. Expected SHA-256: {expected_digest}. "
        f"Release URL: {entry['url']}. Recovery: run "
        f"`{_DOWNLOAD_COMMAND.format(name=entry['name'])}` or set "
        f"{entry['legacy_override_env']} to an exact verified checkpoint."
    )


def _stat_identity(stat_result: os.stat_result) -> tuple[int, int, int]:
    return (stat_result.st_dev, stat_result.st_ino, stat_result.st_ctime_ns)


def _stat_key(
    path: Path, stat_result: os.stat_result, expected_sha256: str
) -> tuple[str, int, int, str]:
    return (
        str(path.resolve()),
        stat_result.st_size,
        stat_result.st_mtime_ns,
        expected_sha256,
    )


def _sha256(path: Path, expected_sha256: str) -> str:
    """Hash a stable path, caching only a matching expected digest."""

    for _attempt in range(3):
        path_before = path.stat()
        key = _stat_key(path, path_before, expected_sha256)
        identity = _stat_identity(path_before)
        if _VERIFIED_HASHES.get(key) == identity:
            return expected_sha256

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            opened_before = os.fstat(stream.fileno())
            if (
                _stat_identity(opened_before) != identity
                or opened_before.st_size != path_before.st_size
                or opened_before.st_mtime_ns != path_before.st_mtime_ns
            ):
                continue
            for chunk in iter(lambda: stream.read(_STREAM_CHUNK_BYTES), b""):
                digest.update(chunk)
            opened_after = os.fstat(stream.fileno())

        path_after = path.stat()
        stable = (
            _stat_identity(opened_before)
            == _stat_identity(opened_after)
            == _stat_identity(path_after)
            and opened_before.st_size == opened_after.st_size == path_after.st_size
            and opened_before.st_mtime_ns
            == opened_after.st_mtime_ns
            == path_after.st_mtime_ns
        )
        if not stable:
            continue

        actual = digest.hexdigest()
        if actual == expected_sha256:
            _VERIFIED_HASHES[_stat_key(path, path_after, expected_sha256)] = (
                _stat_identity(path_after)
            )
        return actual

    raise OSError(f"file changed while hashing: {path}")


def _remember_verified(path: Path, expected_sha256: str) -> None:
    stat_result = path.stat()
    _VERIFIED_HASHES[_stat_key(path, stat_result, expected_sha256)] = _stat_identity(
        stat_result
    )


def _resolved(
    path: Path,
    entry: dict[str, Any],
    *,
    expected: str,
    format_name: str,
    is_legacy: bool,
) -> ResolvedModelAsset:
    try:
        actual = _sha256(path, expected)
    except OSError as error:
        raise ModelAssetError(
            _error_message(entry, f"could not be verified ({error})", location=path, expected=expected)
        ) from error
    if actual != expected:
        raise ModelAssetError(
            _error_message(
                entry,
                f"has SHA-256 {actual}, which does not match",
                location=path,
                expected=expected,
            )
        )
    return ResolvedModelAsset(
        path=path.resolve(),
        sha256=actual,
        format=format_name,
        is_legacy=is_legacy,
        manifest_entry=entry,
    )


def _resolve_override(path: Path, entry: dict[str, Any]) -> ResolvedModelAsset:
    if path.suffix == ".safetensors":
        expected = entry["sha256"]
        format_name = "safetensors"
        is_legacy = False
    elif path.suffix == ".pt":
        expected = entry["parent"]["sha256"]
        format_name = "pytorch"
        is_legacy = True
    else:
        raise ModelAssetError(
            _error_message(
                entry,
                "uses an unsupported override suffix; expected .safetensors or .pt",
                location=path,
            )
        )
    if not path.is_file():
        raise ModelAssetError(
            _error_message(entry, "override is missing or not a file", location=path, expected=expected)
        )
    return _resolved(
        path,
        entry,
        expected=expected,
        format_name=format_name,
        is_legacy=is_legacy,
    )


def _offline() -> bool:
    return os.environ.get("TENNETSAC_OFFLINE") == "1"


def _is_https(url: object) -> bool:
    if not isinstance(url, str):
        return False
    parsed = urlsplit(url)
    return parsed.scheme.lower() == "https" and bool(parsed.netloc)


def _close_response(response: object) -> None:
    close = getattr(response, "close", None)
    if close is not None:
        try:
            close()
        except Exception:
            pass


def _download_asset(
    entry: dict[str, Any], target: Path
) -> ResolvedModelAsset:
    configured_url = entry["url"]
    if not _is_https(configured_url):
        raise ModelAssetError(
            _error_message(
                entry,
                "configured download URL is not HTTPS",
                location=configured_url,
            )
        )

    try:
        response = urlopen(configured_url)
    except Exception as error:
        raise ModelAssetError(
            _error_message(
                entry,
                f"could not open the HTTPS release download ({error})",
                location=configured_url,
            )
        ) from error

    partial = target.with_name(
        f"{target.name}.part.{os.getpid()}.{uuid4().hex}"
    )
    try:
        try:
            final_url = response.geturl()
        except Exception as error:
            raise ModelAssetError(
                _error_message(
                    entry,
                    f"could not determine the final download URL ({error})",
                    location=configured_url,
                )
            ) from error
        if not _is_https(final_url):
            raise ModelAssetError(
                _error_message(
                    entry,
                    "final redirected download URL is not HTTPS",
                    location=final_url,
                )
            )

        digest = hashlib.sha256()
        try:
            with partial.open("xb") as stream:
                while True:
                    chunk = response.read(_STREAM_CHUNK_BYTES)
                    if not chunk:
                        break
                    stream.write(chunk)
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as error:
            raise ModelAssetError(
                _error_message(
                    entry,
                    f"download could not be written atomically ({error})",
                    location=target,
                )
            ) from error

        actual = digest.hexdigest()
        if actual != entry["sha256"]:
            raise ModelAssetError(
                _error_message(
                    entry,
                    f"download has SHA-256 {actual}, which does not match",
                    location=target,
                )
            )

        try:
            os.replace(partial, target)
            _remember_verified(target, entry["sha256"])
        except Exception as error:
            raise ModelAssetError(
                _error_message(
                    entry,
                    f"verified download could not be installed ({error})",
                    location=target,
                )
            ) from error
        return ResolvedModelAsset(
            path=target.resolve(),
            sha256=actual,
            format=entry["format"],
            is_legacy=False,
            manifest_entry=entry,
        )
    finally:
        _close_response(response)
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass


def _resolve_cache_under_lock(
    entry: dict[str, Any], target: Path, *, allow_download: bool
) -> ResolvedModelAsset:
    lock_path = target.with_name(f"{target.name}.lock")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ModelAssetError(
            _error_message(
                entry,
                f"cache directory could not be created ({error})",
                location=target,
            )
        ) from error

    try:
        with FileLock(str(lock_path), timeout=_LOCK_TIMEOUT_SECONDS):
            corrupt_error = None
            if target.is_file():
                try:
                    return _resolved(
                        target,
                        entry,
                        expected=entry["sha256"],
                        format_name=entry["format"],
                        is_legacy=False,
                    )
                except ModelAssetError as error:
                    corrupt_error = error
                    try:
                        target.unlink()
                    except OSError as unlink_error:
                        raise ModelAssetError(
                            _error_message(
                                entry,
                                f"corrupt cache file could not be removed ({unlink_error})",
                                location=target,
                            )
                        ) from unlink_error

            if _offline() or not allow_download:
                reason = "is not available locally and network access is disabled"
                if corrupt_error is not None:
                    reason = f"was corrupt and was removed ({corrupt_error})"
                raise ModelAssetError(
                    _error_message(entry, reason, location=target)
                )

            return _download_asset(entry, target)
    except Timeout as error:
        raise ModelAssetError(
            _error_message(
                entry,
                f"timed out waiting for lock {lock_path} protecting cache target",
                location=target,
            )
        ) from error
    except ModelAssetError:
        raise
    except Exception as error:
        raise ModelAssetError(
            _error_message(
                entry,
                f"cache lock {lock_path} failed ({error})",
                location=target,
            )
        ) from error


def resolve_model_asset(
    name: str = "smi-ted-light", allow_download: bool = True
) -> ResolvedModelAsset:
    entry = external_model(name)
    if name != "smi-ted-light":
        raise ModelAssetError(f"External model {name!r} is not managed by this resolver")

    override_value = os.environ.get(entry["legacy_override_env"])
    if override_value:
        return _resolve_override(Path(override_value).expanduser(), entry)

    target = asset_cache_path(entry)
    if target.is_file():
        try:
            return _resolved(
                target,
                entry,
                expected=entry["sha256"],
                format_name=entry["format"],
                is_legacy=False,
            )
        except ModelAssetError:
            return _resolve_cache_under_lock(
                entry, target, allow_download=allow_download
            )

    if _offline() or not allow_download:
        raise ModelAssetError(
            _error_message(
                entry,
                "is not available locally and network access is disabled",
                location=target,
            )
        )
    return _resolve_cache_under_lock(entry, target, allow_download=allow_download)


def verify_model_asset(name: str = "smi-ted-light") -> ResolvedModelAsset:
    return resolve_model_asset(name, allow_download=False)


__all__ = [
    "ModelAssetError",
    "ResolvedModelAsset",
    "asset_cache_path",
    "resolve_model_asset",
    "verify_model_asset",
]
