#!/usr/bin/env python3
"""Turn a served pack into a Hugging Face model repo that serves itself: <pack>/recipe (this recipe, the
tracked files of this checkout) and <pack>/README.md (the model card, from tools/pack_readme.py).

Whoever downloads the repo runs
    hf download ORG/NAME --local-dir ~/models/NAME
    cd ~/models/NAME/recipe && ./serve.sh
and the recipe takes the pack as its parent directory: nothing to convert, diffbot's recipe/ and image are
fetched/built on the first run, the K=2 kernels built, the Engram shards fetched if the repo leaves them out.

usage: make_hf_pack.py <pack> --repo ORG/NAME [--no-engram] [--force-readme] [--status TEXT]
Prints the upload command (hf upload-large-folder: resumable, parallel) with the exclusions for local state.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pack_readme  # noqa: E402
import resolve_model  # noqa: E402

ENGRAM_SHARDS = ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")
# local state a recipe folder accumulates (nothing of it belongs in the repo)
LOCAL_STATE = (".env", ".cache", "models", "diffbot", "drafter-1017", "kernels/tree", "__pycache__", "logs")
PACK_EXCLUDES = (".cache/*", "config.json.pre-rewrite", "pack_scan.json")


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit(f"{ROOT} is not a git checkout of the recipe: cannot tell its files from local state")
    return [p for p in out.decode().split("\0") if p and (ROOT / p).is_file()]


def dirty() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True).stdout.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--repo", required=True, help="Hugging Face repo id ORG/NAME")
    ap.add_argument("--no-engram", action="store_true", help="leave diffbot's Engram shards 47/48 out of the upload")
    ap.add_argument("--force-readme", action="store_true", help="overwrite an existing README.md")
    ap.add_argument("--status", default=pack_readme.DEFAULT_STATUS, help="status line of the model card")
    a = ap.parse_args()
    pack = Path(a.pack).resolve()
    if not resolve_model.HF_RE.match(a.repo):
        raise SystemExit(f"--repo {a.repo!r} is not ORG/NAME")
    if not resolve_model.is_served_pack(pack):
        raise SystemExit(f"{pack} is not a served pack (run ./prepare_pack.py first; the raw quant is not what gets hosted)")
    ok, why = resolve_model.quant_complete(pack)
    if not ok:
        raise SystemExit(f"{pack}: {why}")

    # recipe/
    files = tracked_files()
    dest = pack / "recipe"
    if dest.resolve() == ROOT.resolve():
        print(f"recipe already lives in the pack ({dest}); nothing to copy")
    else:
        d = dirty()
        if d:
            print(f"note: this checkout has uncommitted changes (copied as they are):\n{d}")
        dest.mkdir(exist_ok=True)
        for rel in files:
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, out)
        stale = []
        for p in dest.rglob("*"):
            rel = p.relative_to(dest).as_posix()
            if p.is_file() and rel not in files and not any(rel == s or rel.startswith(s + "/") or "/__pycache__/" in f"/{rel}" for s in LOCAL_STATE):
                stale.append(rel)
        print(f"recipe: {len(files)} files -> {dest}")
        if stale:
            print(f"note: {len(stale)} files in {dest} are not part of the recipe (left alone; exclude or delete them): "
                  + ", ".join(stale[:8]) + (" ..." if len(stale) > 8 else ""))

    # README.md
    readme = pack / "README.md"
    if readme.is_file() and not a.force_readme:
        print(f"README.md kept ({readme}); --force-readme regenerates it")
    else:
        readme.write_text(pack_readme.render(a.repo, pack, a.status, not a.no_engram))
        print(f"README.md written ({readme}); edit it")

    # upload command
    excludes = list(PACK_EXCLUDES) + [f"recipe/{s}" if s in (".env",) else f"recipe/{s}/*" for s in LOCAL_STATE]
    excludes += ["recipe/**/__pycache__/*", "recipe/**/*.pyc"]
    if a.no_engram:
        excludes += list(ENGRAM_SHARDS)
    ex = " ".join(f'"{e}"' for e in excludes)
    print("\nupload (resumable; re-run after an interruption; the repo is created if it does not exist, add --private to keep it so):")
    print(f"  hf upload-large-folder {a.repo} '{pack}' --repo-type model --num-workers 8 --exclude {ex}")
    print(f"\nthen anyone serves it with:\n  hf download {a.repo} --local-dir ~/models/{a.repo.split('/')[1]} --max-workers 8\n"
          f"  cd ~/models/{a.repo.split('/')[1]}/recipe && ./serve.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
