"""Immutable, streamed downloads with a locally pinned SHA-256 manifest."""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, target, manifest_path, attempts=3):
    target, manifest_path = Path(target), Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {"artifacts": []}
    )
    records = [a for a in manifest["artifacts"] if a["filename"] == target.as_posix()]
    if target.exists():
        if (
            len(records) != 1
            or records[0]["source"] != url
            or sha256(target) != records[0]["sha256"]
        ):
            raise ValueError("Existing raw file is unregistered or changed; refusing overwrite")
        return records[0]
    if records:
        expected = records[0]["sha256"]
    else:
        expected = None
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(attempts):
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError("Download returned HTML instead of market data")
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            digest = sha256(temporary)
            if expected and digest != expected:
                raise ValueError("Downloaded artifact does not match pinned SHA-256")
            temporary.rename(target)
            record = {
                "filename": target.as_posix(),
                "source": url,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "size_bytes": target.stat().st_size,
                "sha256": digest,
            }
            if not records:
                manifest["artifacts"].append(record)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            return records[0] if records else record
        except Exception:
            logging.exception("Download attempt %s/%s failed: %s", attempt + 1, attempts, url)
            temporary.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
