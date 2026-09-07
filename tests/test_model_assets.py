import hashlib
import os
from pathlib import Path

import pytest

from tennetsac import model_manifest
from tennetsac import _model_assets as assets
from tennetsac._model_assets import (
    ModelAssetError,
    ResolvedModelAsset,
    asset_cache_path,
    resolve_model_asset,
    verify_model_asset,
)


@pytest.fixture(autouse=True)
def isolate_asset_environment(monkeypatch):
    monkeypatch.delenv("TENNETSAC_CACHE_DIR", raising=False)
    monkeypatch.delenv("TENNETSAC_OFFLINE", raising=False)
    monkeypatch.delenv("TENNETSAC_SMI_TED_CHECKPOINT", raising=False)
    assets._clear_verified_hash_cache()
    yield
    assets._clear_verified_hash_cache()


@pytest.fixture
def smi_entry():
    return model_manifest.external_model("smi-ted-light")


def _entry_for_bytes(smi_entry, data: bytes) -> dict:
    entry = dict(smi_entry)
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    return entry


def _configure_entry(monkeypatch, tmp_path, entry):
    monkeypatch.setenv("TENNETSAC_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(assets, "external_model", lambda name: entry)


def _assert_actionable_error(error, entry, location, expected=None):
    message = str(error)
    assert entry["name"] in message
    assert entry["release_tag"] in message
    assert entry["url"] in message
    assert str(location) in message
    assert (expected or entry["sha256"]) in message
    assert f"python -m tennetsac.model_assets download {entry['name']}" in message


class FakeHTTPSResponse:
    def __init__(self, chunks, *, final_url="https://downloads.example/model", headers=None):
        self._chunks = iter(chunks)
        self._final_url = final_url
        self.headers = {} if headers is None else headers
        self.closed = False

    def geturl(self):
        return self._final_url

    def read(self, _size):
        value = next(self._chunks, b"")
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self):
        self.closed = True


def test_cache_path_uses_versioned_model_directory(monkeypatch, tmp_path, smi_entry):
    monkeypatch.setenv("TENNETSAC_CACHE_DIR", str(tmp_path))

    assert asset_cache_path(smi_entry) == (
        tmp_path
        / "models"
        / "smi-ted-light"
        / "v1"
        / "smi-ted-light-inference-v1.safetensors"
    )


def test_cache_path_uses_platform_default_when_override_is_absent(
    monkeypatch, tmp_path, smi_entry
):
    monkeypatch.delenv("TENNETSAC_CACHE_DIR", raising=False)
    monkeypatch.setattr(assets, "user_cache_path", lambda name: tmp_path / name)

    assert asset_cache_path(smi_entry) == (
        tmp_path
        / "tennetsac"
        / "models"
        / "smi-ted-light"
        / "v1"
        / "smi-ted-light-inference-v1.safetensors"
    )


def test_explicit_override_wins_and_is_not_modified(monkeypatch, tmp_path, smi_entry):
    override = tmp_path / "user-owned.safetensors"
    contents = b"verified override"
    override.write_bytes(contents)
    entry = _entry_for_bytes(smi_entry, contents)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("TENNETSAC_CACHE_DIR", str(cache_root))
    monkeypatch.setenv(entry["legacy_override_env"], str(override))
    monkeypatch.setattr(assets, "external_model", lambda name: entry)

    resolved = resolve_model_asset()

    assert resolved.path == override.resolve()
    assert resolved.sha256 == entry["sha256"]
    assert resolved.format == "safetensors"
    assert resolved.is_legacy is False
    assert resolved.manifest_entry == entry
    assert override.read_bytes() == contents
    assert not cache_root.exists()


def test_offline_cache_miss_is_actionable_and_never_opens_network(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TENNETSAC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TENNETSAC_OFFLINE", "1")
    network_calls = []
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: network_calls.append(1))

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    message = str(captured.value)
    assert "smi-ted-light" in message
    assert "model-smi-ted-light-v1" in message
    assert "https://github.com/" in message
    assert str(asset_cache_path(model_manifest.external_model("smi-ted-light"))) in message
    assert "566c828ab592a4bfd9050906e4d7f64273a9a27517bef175b00dbed38eec94fc" in message
    assert "python -m tennetsac.model_assets download smi-ted-light" in message
    assert "TENNETSAC_SMI_TED_CHECKPOINT" in message
    assert network_calls == []


def test_valid_download_streams_to_verified_atomic_cache(
    monkeypatch, tmp_path, smi_entry
):
    chunks = [b"streamed ", b"asset", b""]
    entry = _entry_for_bytes(smi_entry, b"streamed asset")
    _configure_entry(monkeypatch, tmp_path, entry)
    response = FakeHTTPSResponse(chunks, headers={"Content-Length": "2"})
    monkeypatch.setattr(assets, "urlopen", lambda url: response)

    resolved = resolve_model_asset()

    target = asset_cache_path(entry)
    assert resolved.path == target.resolve()
    assert resolved.sha256 == entry["sha256"]
    assert resolved.format == "safetensors"
    assert resolved.is_legacy is False
    assert resolved.manifest_entry == entry
    assert target.read_bytes() == b"streamed asset"
    assert response.closed
    assert list(target.parent.glob("*.part.*")) == []


def test_configured_non_https_url_is_rejected_before_network(
    monkeypatch, tmp_path, smi_entry
):
    entry = _entry_for_bytes(smi_entry, b"asset")
    entry["url"] = "http://downloads.example/model"
    _configure_entry(monkeypatch, tmp_path, entry)
    calls = []
    monkeypatch.setattr(assets, "urlopen", lambda url: calls.append(url))

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, entry["url"])
    assert "HTTPS" in str(captured.value)
    assert calls == []


def test_final_non_https_redirect_is_rejected(monkeypatch, tmp_path, smi_entry):
    entry = _entry_for_bytes(smi_entry, b"asset")
    _configure_entry(monkeypatch, tmp_path, entry)
    final_url = "http://downloads.example/model"
    response = FakeHTTPSResponse([b"asset", b""], final_url=final_url)
    monkeypatch.setattr(assets, "urlopen", lambda url: response)

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, final_url)
    assert "HTTPS" in str(captured.value)
    assert response.closed
    assert not asset_cache_path(entry).exists()


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "0"}, {"Content-Length": "5"}])
def test_content_length_never_substitutes_for_sha256(
    monkeypatch, tmp_path, smi_entry, headers
):
    entry = _entry_for_bytes(smi_entry, b"right")
    _configure_entry(monkeypatch, tmp_path, entry)
    response = FakeHTTPSResponse([b"wrong", b""], headers=headers)
    monkeypatch.setattr(assets, "urlopen", lambda url: response)

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    target = asset_cache_path(entry)
    _assert_actionable_error(captured.value, entry, target)
    assert hashlib.sha256(b"wrong").hexdigest() in str(captured.value)
    assert not target.exists()
    assert list(target.parent.glob("*.part.*")) == []


def test_interrupted_read_removes_only_its_partial_and_preserves_cause(
    monkeypatch, tmp_path, smi_entry
):
    entry = _entry_for_bytes(smi_entry, b"complete")
    _configure_entry(monkeypatch, tmp_path, entry)
    target = asset_cache_path(entry)
    target.parent.mkdir(parents=True)
    unrelated_partial = target.parent / f"{target.name}.part.other-process"
    unrelated_partial.write_bytes(b"owned elsewhere")
    read_error = OSError("connection interrupted")
    response = FakeHTTPSResponse([b"incom", read_error])
    monkeypatch.setattr(assets, "urlopen", lambda url: response)

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, target)
    assert captured.value.__cause__ is read_error
    assert not target.exists()
    assert unrelated_partial.read_bytes() == b"owned elsewhere"
    assert list(target.parent.glob(f"{target.name}.part.*")) == [unrelated_partial]


def test_write_failure_removes_partial_and_preserves_cause(
    monkeypatch, tmp_path, smi_entry
):
    data = b"download"
    entry = _entry_for_bytes(smi_entry, data)
    _configure_entry(monkeypatch, tmp_path, entry)
    target = asset_cache_path(entry)
    response = FakeHTTPSResponse([data, b""])
    monkeypatch.setattr(assets, "urlopen", lambda url: response)
    real_open = Path.open
    write_error = OSError("disk full")

    class FailingWriter:
        def __init__(self, stream):
            self._stream = stream

        def __enter__(self):
            self._stream.__enter__()
            return self

        def __exit__(self, *args):
            return self._stream.__exit__(*args)

        def write(self, _chunk):
            raise write_error

        def flush(self):
            return self._stream.flush()

        def fileno(self):
            return self._stream.fileno()

    def fail_partial_write(path, *args, **kwargs):
        stream = real_open(path, *args, **kwargs)
        return FailingWriter(stream) if ".part." in path.name else stream

    monkeypatch.setattr(Path, "open", fail_partial_write)

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, target)
    assert captured.value.__cause__ is write_error
    assert not target.exists()
    assert list(target.parent.glob("*.part.*")) == []


def test_fsync_failure_removes_partial_and_preserves_cause(
    monkeypatch, tmp_path, smi_entry
):
    data = b"download"
    entry = _entry_for_bytes(smi_entry, data)
    _configure_entry(monkeypatch, tmp_path, entry)
    target = asset_cache_path(entry)
    monkeypatch.setattr(
        assets, "urlopen", lambda url: FakeHTTPSResponse([data, b""])
    )
    fsync_error = OSError("fsync failed")

    def fail_fsync(_fd):
        raise fsync_error

    monkeypatch.setattr(assets.os, "fsync", fail_fsync)

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, target)
    assert captured.value.__cause__ is fsync_error
    assert not target.exists()
    assert list(target.parent.glob("*.part.*")) == []


def test_lock_timeout_names_lock_and_target_and_preserves_cause(
    monkeypatch, tmp_path, smi_entry
):
    entry = _entry_for_bytes(smi_entry, b"asset")
    _configure_entry(monkeypatch, tmp_path, entry)
    target = asset_cache_path(entry)
    lock_path = target.with_name(f"{target.name}.lock")
    timeout = assets.Timeout(str(lock_path))

    class TimedOutLock:
        def __enter__(self):
            raise timeout

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(assets, "FileLock", lambda *args, **kwargs: TimedOutLock())

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, target)
    assert str(lock_path) in str(captured.value)
    assert captured.value.__cause__ is timeout


def test_lock_io_failure_is_actionable_and_preserves_cause(
    monkeypatch, tmp_path, smi_entry
):
    entry = _entry_for_bytes(smi_entry, b"asset")
    _configure_entry(monkeypatch, tmp_path, entry)
    target = asset_cache_path(entry)
    lock_path = target.with_name(f"{target.name}.lock")
    lock_error = OSError("lock filesystem unavailable")

    class BrokenLock:
        def __enter__(self):
            raise lock_error

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(assets, "FileLock", lambda *args, **kwargs: BrokenLock())

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, target)
    assert str(lock_path) in str(captured.value)
    assert captured.value.__cause__ is lock_error


def test_lock_recheck_uses_asset_written_by_another_process(
    monkeypatch, tmp_path, smi_entry
):
    data = b"won by another process"
    entry = _entry_for_bytes(smi_entry, data)
    _configure_entry(monkeypatch, tmp_path, entry)
    target = asset_cache_path(entry)

    class CompletingLock:
        def __enter__(self):
            target.write_bytes(data)
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(assets, "FileLock", lambda *args, **kwargs: CompletingLock())
    monkeypatch.setattr(
        assets,
        "urlopen",
        lambda url: pytest.fail("lock recheck must avoid a duplicate download"),
    )

    resolved = resolve_model_asset()

    assert resolved.path == target.resolve()
    assert target.read_bytes() == data


def test_corrupt_cache_is_replaced_under_lock(monkeypatch, tmp_path, smi_entry):
    data = b"replacement"
    entry = _entry_for_bytes(smi_entry, data)
    _configure_entry(monkeypatch, tmp_path, entry)
    target = asset_cache_path(entry)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt")
    monkeypatch.setattr(
        assets, "urlopen", lambda url: FakeHTTPSResponse([data, b""])
    )

    resolved = resolve_model_asset()

    assert resolved.sha256 == entry["sha256"]
    assert target.read_bytes() == data


def test_online_verification_io_failure_preserves_cache_and_cause_without_download(
    monkeypatch, tmp_path, smi_entry
):
    data = b"valid cache bytes"
    entry = _entry_for_bytes(smi_entry, data)
    _configure_entry(monkeypatch, tmp_path, entry)
    target = asset_cache_path(entry)
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    verification_error = OSError("cache read failed")
    download_calls = []

    def fail_verification(_path, _expected):
        raise verification_error

    def unexpected_download(*args, **kwargs):
        download_calls.append((args, kwargs))
        raise ModelAssetError("download must not run after incomplete verification")

    monkeypatch.setattr(assets, "_sha256", fail_verification)
    monkeypatch.setattr(assets, "_download_asset", unexpected_download)

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, target)
    assert captured.value.__cause__ is verification_error
    assert target.read_bytes() == data
    assert download_calls == []


def test_offline_verification_io_failure_preserves_cache_and_cause_without_network(
    monkeypatch, tmp_path, smi_entry
):
    data = b"valid cache bytes"
    entry = _entry_for_bytes(smi_entry, data)
    _configure_entry(monkeypatch, tmp_path, entry)
    monkeypatch.setenv("TENNETSAC_OFFLINE", "1")
    target = asset_cache_path(entry)
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    verification_error = OSError("cache stat failed")
    network_calls = []

    def fail_verification(_path, _expected):
        raise verification_error

    monkeypatch.setattr(assets, "_sha256", fail_verification)
    monkeypatch.setattr(
        assets, "urlopen", lambda *args, **kwargs: network_calls.append((args, kwargs))
    )

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, target)
    assert captured.value.__cause__ is verification_error
    assert target.read_bytes() == data
    assert network_calls == []


def test_offline_corrupt_cache_is_removed_without_network(
    monkeypatch, tmp_path, smi_entry
):
    entry = _entry_for_bytes(smi_entry, b"expected")
    _configure_entry(monkeypatch, tmp_path, entry)
    monkeypatch.setenv("TENNETSAC_OFFLINE", "1")
    target = asset_cache_path(entry)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt")
    monkeypatch.setattr(
        assets, "urlopen", lambda url: pytest.fail("offline mode opened the network")
    )

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, target)
    assert "removed" in str(captured.value).lower()
    assert not target.exists()


def test_invalid_user_override_is_rejected_but_never_removed(
    monkeypatch, tmp_path, smi_entry
):
    entry = _entry_for_bytes(smi_entry, b"expected")
    _configure_entry(monkeypatch, tmp_path / "cache", entry)
    override = tmp_path / "user-owned.safetensors"
    override.write_bytes(b"wrong")
    monkeypatch.setenv(entry["legacy_override_env"], str(override))

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, override)
    assert override.read_bytes() == b"wrong"


def test_verified_hash_cache_rehashes_replacement_with_same_size_and_mtime(
    monkeypatch, tmp_path, smi_entry
):
    original = b"right"
    replacement = b"wrong"
    entry = _entry_for_bytes(smi_entry, original)
    _configure_entry(monkeypatch, tmp_path / "cache", entry)
    override = tmp_path / "user-owned.safetensors"
    override.write_bytes(original)
    monkeypatch.setenv(entry["legacy_override_env"], str(override))
    original_mtime_ns = override.stat().st_mtime_ns
    resolve_model_asset()

    staged = tmp_path / "replacement.safetensors"
    staged.write_bytes(replacement)
    os.replace(staged, override)
    os.utime(override, ns=(original_mtime_ns, original_mtime_ns))

    with pytest.raises(ModelAssetError):
        resolve_model_asset()

    assert override.read_bytes() == replacement


def test_matching_asset_is_hashed_only_once_for_an_unchanged_stat_fingerprint(
    monkeypatch, tmp_path, smi_entry
):
    contents = b"unchanged"
    entry = _entry_for_bytes(smi_entry, contents)
    _configure_entry(monkeypatch, tmp_path / "cache", entry)
    override = tmp_path / "user-owned.safetensors"
    override.write_bytes(contents)
    monkeypatch.setenv(entry["legacy_override_env"], str(override))
    real_sha256 = hashlib.sha256
    hash_calls = []

    def counting_sha256(*args, **kwargs):
        hash_calls.append(1)
        return real_sha256(*args, **kwargs)

    monkeypatch.setattr(assets.hashlib, "sha256", counting_sha256)

    resolve_model_asset()
    resolve_model_asset()

    assert hash_calls == [1]


def test_https_open_failure_is_actionable_and_preserves_cause(
    monkeypatch, tmp_path, smi_entry
):
    entry = _entry_for_bytes(smi_entry, b"asset")
    _configure_entry(monkeypatch, tmp_path, entry)
    network_error = OSError("TLS connection failed")

    def fail_open(_url):
        raise network_error

    monkeypatch.setattr(assets, "urlopen", fail_open)

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, entry["url"])
    assert captured.value.__cause__ is network_error


def test_exact_parent_digest_resolves_pt_override_as_legacy(
    monkeypatch, tmp_path, smi_entry
):
    contents = b"legacy parent"
    entry = dict(smi_entry)
    entry["parent"] = dict(smi_entry["parent"])
    entry["parent"]["sha256"] = hashlib.sha256(contents).hexdigest()
    _configure_entry(monkeypatch, tmp_path / "cache", entry)
    override = tmp_path / "parent.pt"
    override.write_bytes(contents)
    monkeypatch.setenv(entry["legacy_override_env"], str(override))

    resolved = resolve_model_asset()

    assert resolved == ResolvedModelAsset(
        path=override.resolve(),
        sha256=entry["parent"]["sha256"],
        format="pytorch",
        is_legacy=True,
        manifest_entry=entry,
    )


def test_symlinked_pt_override_preserves_lexical_suffix_for_legacy_dispatch(
    monkeypatch, tmp_path, smi_entry
):
    contents = b"legacy parent behind a suffixless blob"
    entry = dict(smi_entry)
    entry["parent"] = dict(smi_entry["parent"])
    entry["parent"]["sha256"] = hashlib.sha256(contents).hexdigest()
    _configure_entry(monkeypatch, tmp_path / "cache", entry)
    blob = tmp_path / entry["parent"]["sha256"]
    blob.write_bytes(contents)
    override = tmp_path / "smi-ted-Light_40.pt"
    override.symlink_to(blob.name)
    original_link_target = override.readlink()
    monkeypatch.setenv(entry["legacy_override_env"], str(override))

    resolved = resolve_model_asset()

    assert resolved.path == override
    assert resolved.path.suffix == ".pt"
    assert resolved.format == "pytorch"
    assert resolved.is_legacy is True
    assert resolved.sha256 == entry["parent"]["sha256"]
    assert override.is_symlink()
    assert override.readlink() == original_link_target
    assert override.read_bytes() == contents


def test_symlinked_safetensors_override_preserves_lexical_suffix_for_dispatch(
    monkeypatch, tmp_path, smi_entry
):
    contents = b"derived asset behind a suffixless blob"
    entry = _entry_for_bytes(smi_entry, contents)
    _configure_entry(monkeypatch, tmp_path / "cache", entry)
    blob = tmp_path / entry["sha256"]
    blob.write_bytes(contents)
    override = tmp_path / "smi-ted-light-inference-v1.safetensors"
    override.symlink_to(blob.name)
    original_link_target = override.readlink()
    monkeypatch.setenv(entry["legacy_override_env"], str(override))

    resolved = resolve_model_asset()

    assert resolved.path == override
    assert resolved.path.suffix == ".safetensors"
    assert resolved.format == "safetensors"
    assert resolved.is_legacy is False
    assert resolved.sha256 == entry["sha256"]
    assert override.is_symlink()
    assert override.readlink() == original_link_target
    assert override.read_bytes() == contents


def test_relative_override_preserves_dotdot_after_symlinked_directory(
    monkeypatch, tmp_path, smi_entry
):
    trusted = b"verified through the symlinked directory"
    untrusted = b"different bytes at the normalized path"
    entry = _entry_for_bytes(smi_entry, trusted)
    _configure_entry(monkeypatch, tmp_path / "cache", entry)

    base = tmp_path / "base"
    base.mkdir()
    elsewhere = tmp_path / "elsewhere"
    symlink_target = elsewhere / "nested"
    symlink_target.mkdir(parents=True)
    symlinked_directory = base / "symlink-dir"
    symlinked_directory.symlink_to(symlink_target, target_is_directory=True)
    original_link_target = symlinked_directory.readlink()

    trusted_path = elsewhere / "file.safetensors"
    trusted_path.write_bytes(trusted)
    normalized_path = base / "file.safetensors"
    normalized_path.write_bytes(untrusted)
    relative_override = Path("base/symlink-dir/../file.safetensors")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(entry["legacy_override_env"], str(relative_override))

    resolved = resolve_model_asset()

    assert resolved.path == tmp_path / relative_override
    assert resolved.path.is_absolute()
    assert ".." in resolved.path.parts
    assert resolved.path.suffix == ".safetensors"
    assert resolved.path.read_bytes() == trusted
    assert resolved.sha256 == entry["sha256"]
    assert resolved.format == "safetensors"
    assert resolved.is_legacy is False
    assert resolved.manifest_entry == entry
    assert symlinked_directory.is_symlink()
    assert symlinked_directory.readlink() == original_link_target
    assert trusted_path.read_bytes() == trusted
    assert normalized_path.read_bytes() == untrusted


@pytest.mark.parametrize("suffix", [".bin", ".PT", ""])
def test_unsupported_override_suffix_is_rejected_without_modification(
    monkeypatch, tmp_path, smi_entry, suffix
):
    entry = _entry_for_bytes(smi_entry, b"contents")
    _configure_entry(monkeypatch, tmp_path / "cache", entry)
    override = tmp_path / f"checkpoint{suffix}"
    override.write_bytes(b"contents")
    monkeypatch.setenv(entry["legacy_override_env"], str(override))

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(captured.value, entry, override)
    assert "suffix" in str(captured.value)
    assert override.read_bytes() == b"contents"


def test_legacy_override_requires_exact_parent_digest_and_is_never_removed(
    monkeypatch, tmp_path, smi_entry
):
    _configure_entry(monkeypatch, tmp_path / "cache", smi_entry)
    override = tmp_path / "parent.pt"
    override.write_bytes(b"not the pinned parent")
    monkeypatch.setenv(smi_entry["legacy_override_env"], str(override))

    with pytest.raises(ModelAssetError) as captured:
        resolve_model_asset()

    _assert_actionable_error(
        captured.value, smi_entry, override, smi_entry["parent"]["sha256"]
    )
    assert override.read_bytes() == b"not the pinned parent"


def test_verify_model_asset_disables_download(monkeypatch, tmp_path, smi_entry):
    _configure_entry(monkeypatch, tmp_path, smi_entry)
    monkeypatch.setattr(
        assets,
        "_download_asset",
        lambda *args, **kwargs: pytest.fail("verification attempted a download"),
    )

    with pytest.raises(ModelAssetError) as captured:
        verify_model_asset()

    _assert_actionable_error(captured.value, smi_entry, asset_cache_path(smi_entry))


def test_download_cli_prints_path_and_digest(monkeypatch, tmp_path, capsys, smi_entry):
    from tennetsac import model_assets as cli

    resolved = ResolvedModelAsset(
        path=tmp_path / "model.safetensors",
        sha256=smi_entry["sha256"],
        format="safetensors",
        is_legacy=False,
        manifest_entry=smi_entry,
    )
    calls = []

    def fake_resolve(name, allow_download=True):
        calls.append((name, allow_download))
        return resolved

    monkeypatch.setattr(cli, "resolve_model_asset", fake_resolve)

    exit_code = cli.main(["download", "smi-ted-light"])

    output = capsys.readouterr()
    assert exit_code == 0
    assert output.out == f"path={resolved.path}\nsha256={resolved.sha256}\n"
    assert output.err == ""
    assert calls == [("smi-ted-light", True)]


def test_verify_cli_uses_download_disabled_verifier(
    monkeypatch, tmp_path, capsys, smi_entry
):
    from tennetsac import model_assets as cli

    resolved = ResolvedModelAsset(
        path=tmp_path / "model.safetensors",
        sha256=smi_entry["sha256"],
        format="safetensors",
        is_legacy=False,
        manifest_entry=smi_entry,
    )
    calls = []

    def fake_verify(name):
        calls.append(name)
        return resolved

    monkeypatch.setattr(cli, "verify_model_asset", fake_verify)
    monkeypatch.setattr(
        cli,
        "resolve_model_asset",
        lambda *args, **kwargs: pytest.fail("verify CLI used download resolution"),
    )

    exit_code = cli.main(["verify", "smi-ted-light"])

    output = capsys.readouterr()
    assert exit_code == 0
    assert output.out == f"path={resolved.path}\nsha256={resolved.sha256}\n"
    assert output.err == ""
    assert calls == ["smi-ted-light"]


def test_cli_failure_returns_nonzero_and_prints_actionable_stderr(
    monkeypatch, capsys
):
    from tennetsac import model_assets as cli

    error = ModelAssetError(
        "run python -m tennetsac.model_assets download smi-ted-light"
    )

    def fail(_name):
        raise error

    monkeypatch.setattr(cli, "verify_model_asset", fail)

    exit_code = cli.main(["verify", "smi-ted-light"])

    output = capsys.readouterr()
    assert exit_code != 0
    assert output.out == ""
    assert "python -m tennetsac.model_assets download smi-ted-light" in output.err
