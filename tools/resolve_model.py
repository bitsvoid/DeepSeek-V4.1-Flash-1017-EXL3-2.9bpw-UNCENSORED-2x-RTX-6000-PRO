#!/usr/bin/env python3
"""Where the model is: MODEL=<dir | hf-org/name>, either the raw quant or an already assembled (served) pack.

  a local directory    MODEL=/data/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
  a Hugging Face repo  MODEL=essefree/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw  -> fetched into <recipe>/models/<org>__<name>
  nothing              when this recipe sits inside a served pack (<pack>/recipe, the way the pack is hosted on
                       Hugging Face), the pack is the recipe's parent directory
  QUANT=<dir> (the earlier name of the setting) is accepted as a local directory.

Two kinds of model directory (MODEL_KIND):
  quant  the Mia-layout EXL3 quant as the quantizer wrote it: prepare_pack.py converts a few modules (GPU) and
         assembles the served pack next to it as <quant>-served (PACK overrides; same filesystem = hardlinks)
  pack   a served pack (PACK_INFO.json, config.json in the plugin's vocabulary): nothing to convert, served as
         it is, so PACK is the directory itself
A Hugging Face repo is of unknown kind until it is fetched: resolve again after the download.

DIFFBOT (diffbot's pack download, of which recipe/ is what is used) defaults to <recipe>/diffbot, where
prepare_pack.py fetches their recipe/ when it is missing.

usage: resolve_model.py [--env .env] [--model M] [--pack P] [--shell | --json]
  --shell prints MODEL= MODEL_SOURCE= MODEL_KIND= QUANT= PACK= HF_MODEL_REPO= DIFFBOT= for `eval` in a shell script.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
PACK_MARKER = "PACK_INFO.json"


def read_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def absolute(p: str) -> Path:
    p = os.path.expanduser(p)
    return Path(p) if os.path.isabs(p) else (ROOT / p).resolve()


def is_served_pack(d: Path) -> bool:
    """A pack prepare_pack.py assembled (or a download of one): PACK_INFO.json, or config.json already in the
    plugin's vocabulary (non_routed_exl3.layers keyed by vLLM prefixes, mtp_experts set)."""
    cfg = d / "config.json"
    if not cfg.is_file():
        return False
    if (d / PACK_MARKER).is_file():
        return True
    try:
        q = json.loads(cfg.read_text()).get("quantization_config") or {}
    except (OSError, ValueError):
        return False
    layers = (q.get("non_routed_exl3") or {}).get("layers") or {}
    return "mtp_experts" in q and bool(layers) and all(k.startswith("language_model.") for k in layers)


def diffbot_dir(env: dict[str, str]) -> Path:
    return absolute(env["DIFFBOT"]) if env.get("DIFFBOT") else ROOT / "diffbot"


def resolve(model: str | None, env: dict[str, str], pack: str | None = None) -> dict:
    model = (model or env.get("MODEL") or env.get("QUANT") or "").strip()
    source = ""
    if not model:
        if is_served_pack(ROOT.parent):
            model, source = str(ROOT.parent), "inpack"
        else:
            raise ValueError("MODEL not set: ./serve.sh --model <dir | hf-org/name>, MODEL= in .env, or run the recipe "
                             "from inside a served pack (<pack>/recipe)")
    if source or model.startswith(("/", "./", "../", "~")) or Path(model).is_dir():
        source, quant, repo = source or "local", absolute(model), ""
    elif HF_RE.match(model):
        source, repo = "hf", model
        quant = ROOT / "models" / model.replace("/", "__")
    else:
        raise ValueError(f"MODEL={model!r} is neither an existing directory nor a Hugging Face repo id (org/name)")
    kind = "pack" if is_served_pack(quant) else "quant"
    pack = pack or env.get("PACK") or ""
    if kind == "pack":
        pack_dir = quant
    else:
        pack_dir = absolute(pack) if pack else Path(str(quant).rstrip("/") + "-served")
    return {"MODEL": model, "MODEL_SOURCE": source, "MODEL_KIND": kind, "QUANT": str(quant), "PACK": str(pack_dir),
            "HF_MODEL_REPO": repo, "DIFFBOT": str(diffbot_dir(env))}


def quant_complete(quant: Path) -> tuple[bool, str]:
    idx = quant / "model.safetensors.index.json"
    if not (quant / "config.json").is_file() or not idx.is_file():
        return False, "config.json / model.safetensors.index.json missing"
    wm = json.loads(idx.read_text())["weight_map"]
    missing = sorted({f for f in wm.values() if not (quant / f).is_file()})
    if missing:
        return False, f"{len(missing)} shards named by the index are missing (e.g. {missing[0]})"
    return True, f"{len(set(wm.values()))} shards"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=str(ROOT / ".env"))
    ap.add_argument("--model")
    ap.add_argument("--pack")
    ap.add_argument("--shell", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        r = resolve(a.model, read_env(Path(a.env)), a.pack)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if a.shell:
        for k, v in r.items():
            print(f"{k}='{v}'")
    else:
        print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
