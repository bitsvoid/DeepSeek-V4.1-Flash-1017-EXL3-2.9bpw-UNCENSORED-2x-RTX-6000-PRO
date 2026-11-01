#!/usr/bin/env python3
"""Header-only copies of safetensors shards from a Hugging Face repo (the tests' fixtures: real tensor names,
dtypes and shapes, zero bytes of data). Two range requests per shard, a few hundred KB in all.

usage: fetch_headers.py <org/name> <dest dir> <shard file>... [--revision REV]
  e.g. fetch_headers.py diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 ./eng_diffbot model-00047-of-00048.safetensors model-00048-of-00048.safetensors
       fetch_headers.py deepseek-ai/DeepSeek-V4.1-Flash-1017 ./stock_drafter model-00044-of-00048.safetensors ... --revision 1bae0e1088993b9b82de08f59de5be5a34e42170
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import urllib.request
from pathlib import Path


def fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "fetch_headers/1"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("dest")
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--revision", default="main")
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for s in a.shards:
        url = f"https://huggingface.co/{a.repo}/resolve/{a.revision}/{s}"
        n = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
        hdr = fetch_range(url, 8, 8 + n - 1)
        if len(hdr) != n:
            raise SystemExit(f"{s}: got {len(hdr)} header bytes, expected {n}")
        (dest / s).write_bytes(struct.pack("<Q", n) + hdr)
        print(f"{dest / s}: header {n} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
