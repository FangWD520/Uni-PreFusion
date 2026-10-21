#!/usr/bin/env python3
"""Download and verify the SHA-256-pinned Uni-PreFusion release assets.

Only the Python standard library is required. Existing mismatched files are never
overwritten. A .part file is resumed only with this tool's matching ownership
record; the server must confirm the exact requested Content-Range.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import urllib.error
import urllib.request


CHUNK = 1024 * 1024
PUBLIC_BASE = "https://github.com/FangWD520/Uni-PreFusion/releases/download/v1.0.0/"


class DownloadError(RuntimeError):
    pass


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            value.update(block)
    return value.hexdigest()


def verify(path, asset):
    if path.is_symlink() or not path.is_file():
        raise DownloadError("Required regular file is missing: " + asset["path"])
    if path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
        raise DownloadError("Size or SHA-256 mismatch; file left untouched: " + asset["path"])


def target_path(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or not relative.parts or any(p in ("", ".", "..") for p in relative.parts):
        raise DownloadError("Unsafe manifest path")
    target = root.joinpath(*relative.parts)
    for parent in (target, *target.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise DownloadError("Symlink in asset destination")
    try:
        target.resolve().relative_to(root)
    except ValueError:
        raise DownloadError("Asset destination escapes repository")
    return target


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "uniprefusion_release_assets_v1":
        raise DownloadError("Unrecognized asset manifest")
    assets = manifest.get("assets", [])
    if len(assets) != 5 or len({a["name"] for a in assets}) != 5:
        raise DownloadError("Expected five unique release assets")
    if len({a["path"] for a in assets}) != 5:
        raise DownloadError("Duplicate asset destinations")
    for asset in assets:
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+", asset["name"])
                or asset["url"] != PUBLIC_BASE + asset["name"]
                or type(asset["bytes"]) is not int or not 0 < asset["bytes"] < 2**31
                or not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])):
            raise DownloadError("Invalid release asset specification")
    return assets


def download_asset(root, asset, opener=urllib.request.urlopen):
    """Fetch one asset; opener is injectable for local HTTP contract tests."""
    root = Path(root).resolve()
    target = target_path(root, asset["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify(target, asset)
        return "verified"
    part = target.with_name(target.name + ".part")
    record = target.with_name(target.name + ".part.json")
    lock = target.with_name(target.name + ".download.lock")
    for item in (part, record, lock):
        if item.is_symlink():
            raise DownloadError("Symlink in download cache")
    expected = {"schema": "uniprefusion_download_part_v1", "name": asset["name"],
                "url": asset["url"], "bytes": asset["bytes"], "sha256": asset["sha256"]}
    try:
        lock_handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise DownloadError("Download lock exists; check for another running downloader: " + asset["path"]) from None
    try:
        with lock_handle:
            lock_handle.write(str(os.getpid()) + "\n")
            lock_handle.flush()
            if part.exists() or record.exists():
                if not part.is_file() or not record.is_file():
                    raise DownloadError("Incomplete ownership record; cache left untouched: " + asset["path"])
                if json.loads(record.read_text(encoding="utf-8")) != expected:
                    raise DownloadError("Foreign partial download; cache left untouched: " + asset["path"])
            else:
                with record.open("x", encoding="utf-8") as handle:
                    json.dump(expected, handle, indent=2)
                    handle.write("\n")
                part.touch(exist_ok=False)
            offset = part.stat().st_size
            if offset > asset["bytes"]:
                raise DownloadError("Partial file exceeds expected size; cache left untouched")
            if offset < asset["bytes"]:
                headers = {"User-Agent": "Uni-PreFusion-model-downloader/1.0", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = "bytes=" + str(offset) + "-"
                request = urllib.request.Request(asset["url"], headers=headers)
                try:
                    response = opener(request, timeout=90)
                except (urllib.error.URLError, OSError):
                    raise DownloadError("Connection failed; owned partial download preserved: " + asset["name"]) from None
                with response:
                    status = response.status
                    encoding = response.headers.get("Content-Encoding", "identity").lower()
                    if encoding not in ("identity", ""):
                        raise DownloadError("Unexpected transfer encoding")
                    if status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if not match or tuple(map(int, match.groups())) != (offset, asset["bytes"] - 1, asset["bytes"]):
                            raise DownloadError("Server returned an invalid Content-Range; partial file preserved")
                        mode = "ab"
                    elif status == 200:
                        # Server ignored Range. Only our identified .part is reset.
                        mode, offset = "wb", 0
                    else:
                        raise DownloadError("Unexpected HTTP status " + str(status))
                    length = response.headers.get("Content-Length")
                    if length is not None and (not length.isdecimal() or int(length) != asset["bytes"] - offset):
                        raise DownloadError("Unexpected HTTP response length; partial file preserved")
                    try:
                        with part.open(mode) as handle:
                            received = offset
                            while True:
                                block = response.read(CHUNK)
                                if not block:
                                    break
                                if received + len(block) > asset["bytes"]:
                                    raise DownloadError("Transfer exceeds expected size")
                                handle.write(block)
                                received += len(block)
                            handle.flush()
                            os.fsync(handle.fileno())
                    except (OSError, urllib.error.URLError, http.client.HTTPException):
                        raise DownloadError("Transfer interrupted; owned partial download preserved: " + asset["name"]) from None
            # Size and digest must pass before a final filename is created.
            verify(part, asset)
            if target.exists():
                raise DownloadError("Destination appeared during download; refusing to overwrite")
            # Atomic no-clobber promotion on the same filesystem.
            try:
                os.link(part, target)
            except FileExistsError:
                raise DownloadError("Destination appeared during promotion; refusing to overwrite") from None
            part.unlink()
            record.unlink()
            return "downloaded and verified"
    except (json.JSONDecodeError, UnicodeError):
        raise DownloadError("Unreadable partial ownership record; cache left untouched") from None
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check all five local assets without network access")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        assets = load_manifest(root / "ckpts" / "model_manifest.json")
        for asset in assets:
            if args.verify_only:
                verify(target_path(root, asset["path"]), asset)
                result = "verified"
            else:
                result = download_asset(root, asset)
            print(asset["path"] + ": " + result, flush=True)
    except (DownloadError, OSError, ValueError, KeyError) as exc:
        # Error messages are ours: do not echo redirected/signed URLs or library errors.
        message = str(exc) if isinstance(exc, DownloadError) else "Local file or manifest error; files preserved."
        print("ERROR: " + message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
