#!/usr/bin/env python3
"""Download Objects365-2020 from BAAI data platform.

The file list is public, but getFileSign requires a logged-in BAAI token.
Export BAAI_TOKEN with the browser localStorage data-appToken.accessToken.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API_BASE = "https://data.baai.ac.cn/api/datahub"
BUCKET = "baai-datasets"
DATASET_ID = "11095568931627092"
PREFIX = "fc5f8a7cf75e52911753d726512224d2"


def request_json(method: str, url: str, data: dict | None = None, token: str | None = None) -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {url}: {exc.read()[:300]!r}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URL error for {url}: {exc}") from exc
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def list_page(prefix: str, page: int) -> tuple[list[dict], int]:
    payload = {
        "bucketName": BUCKET,
        "pageSize": 200,
        "pageNumber": page,
        "prefix": prefix,
    }
    out = request_json("POST", f"{API_BASE}/datasetFile/v1/getFileList", payload)
    if out.get("code") != 0:
        raise RuntimeError(f"getFileList failed for {prefix}: {out}")
    data = out.get("data") or {}
    return data.get("elements") or [], int(data.get("total") or 0)


def walk(prefix: str) -> list[dict]:
    files: list[dict] = []
    page = 1
    seen = 0
    total = 1
    while seen < total:
        entries, total = list_page(prefix, page)
        seen += len(entries)
        for entry in entries:
            if entry.get("fileType") == "folder":
                files.extend(walk(entry["filePath"]))
            else:
                files.append(entry)
        page += 1
    return files


def get_signed_url(file_path: str, token: str) -> str:
    payload = {
        "bucketName": BUCKET,
        "datasetId": DATASET_ID,
        "type": 1,
        "filePath": file_path,
    }
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            out = request_json("POST", f"{API_BASE}/datasetFile/v2/getFileSign", payload, token=token)
            break
        except Exception as exc:
            last_error = exc
            wait_s = min(60, attempt * 5)
            print(f"sign retry {attempt}/5 for {file_path}: {exc}; sleep {wait_s}s", flush=True)
            time.sleep(wait_s)
    else:
        raise RuntimeError(f"getFileSign failed after retries for {file_path}: {last_error}")
    if not out:
        raise RuntimeError("getFileSign returned empty response; BAAI_TOKEN is likely missing or expired.")
    if out.get("code") not in (0, None) and "data" not in out:
        raise RuntimeError(f"getFileSign failed for {file_path}: {out}")
    url = out.get("data") if isinstance(out, dict) else None
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise RuntimeError(f"Invalid signed URL for {file_path}: {out}")
    return url.replace("http://", "https://", 1)


def relative_path(file_path: str) -> Path:
    prefix = PREFIX + "/"
    if not file_path.startswith(prefix):
        raise ValueError(f"Unexpected file path: {file_path}")
    return Path(file_path[len(prefix):])


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "-c",
        "--tries=0",
        "--timeout=60",
        "--read-timeout=60",
        "-O",
        str(dst),
        url,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/data/dataset/Objects365_2020")
    parser.add_argument("--parts", nargs="+", default=["license", "train", "val", "test"])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    roots = [f"{PREFIX}/{part.strip('/')}/" for part in args.parts]
    files: list[dict] = []
    for root in roots:
        files.extend(walk(root))
    files.sort(key=lambda x: x["filePath"])
    if args.max_files is not None:
        files = files[: args.max_files]

    manifest = Path(args.manifest) if args.manifest else Path(args.out) / "objects365_baai_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(files, indent=2, ensure_ascii=False), encoding="utf-8")
    total_size = sum(int(f.get("fileSize") or 0) for f in files)
    print(f"manifest={manifest}")
    print(f"files={len(files)} compressed_bytes={total_size}")

    if args.dry_run:
        for f in files[:20]:
            print(relative_path(f["filePath"]), f.get("fileSize"))
        return 0

    token = os.environ.get("BAAI_TOKEN", "").strip()
    if not token:
        print("ERROR: BAAI_TOKEN is required for signed download URLs.", file=sys.stderr)
        print("Open the BAAI page after login, read localStorage['data-appToken'], and export its accessToken.", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    for idx, f in enumerate(files, 1):
        rel = relative_path(f["filePath"])
        dst = out_dir / rel
        expected = int(f.get("fileSize") or 0)
        if expected > 0 and dst.exists() and dst.stat().st_size == expected:
            print(f"[{idx}/{len(files)}] skip complete {rel}")
            continue
        print(f"[{idx}/{len(files)}] signing {rel}")
        signed_url = get_signed_url(f["filePath"], token)
        print(f"[{idx}/{len(files)}] download {rel}")
        download(signed_url, dst)
        time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
