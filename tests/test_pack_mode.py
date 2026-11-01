#!/usr/bin/env python3
"""The served pack as it is hosted on Hugging Face: recipe/ inside the pack, served without conversion.

Builds the header-only served pack with tests/test_offline.py (--keep), then:
  * resolve_model tells the pack from the quant;
  * tools/make_hf_pack.py copies the recipe into <pack>/recipe and writes the model card;
  * with the Engram shards removed from the pack (a repo uploaded without them), <pack>/recipe/prepare_pack.py
    run with no --model takes the pack as its parent, links the shards back from the diffbot folder, installs the
    plan and passes the checks; --skip-download with no Engram anywhere refuses with a clear message;
  * from this checkout, prepare_pack.py --model <pack> does the same (no conversion attempted);
  * resolve_model.py --shell inside the pack prints MODEL_KIND='pack' and PACK = the pack.

usage: test_pack_mode.py --quant-meta <dir> --stock-drafter <dir> --engram <dir>   (as tests/test_offline.py)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import resolve_model  # noqa: E402

ENGRAM = ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")


def sh(*cmd, env=None, check=True) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=check, env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-meta", required=True)
    ap.add_argument("--stock-drafter", required=True)
    ap.add_argument("--engram", required=True)
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="dsv41-29-pack-"))
    r = sh(sys.executable, ROOT / "tests" / "test_offline.py", "--quant-meta", a.quant_meta, "--stock-drafter", a.stock_drafter,
           "--engram", a.engram, "--keep", work, check=False)
    assert r.returncode == 0, f"test_offline failed:\n{r.stdout}\n{r.stderr}"
    work = work.resolve()   # macOS: /var -> /private/var, as resolve_model prints it
    quant, diffbot, pack = work / "quant", work / "diffbot", work / "pack"
    none_env = work / "none.env"

    assert resolve_model.is_served_pack(pack) and not resolve_model.is_served_pack(quant)
    cfg_only = work / "cfg-only"
    cfg_only.mkdir()
    shutil.copy(pack / "config.json", cfg_only / "config.json")
    assert resolve_model.is_served_pack(cfg_only), "a pack is recognised by its config.json alone (PACK_INFO.json left out)"

    # package for Hugging Face
    r = sh(sys.executable, ROOT / "tools" / "make_hf_pack.py", pack, "--repo", "essefree/test-pack", "--no-engram")
    rec = pack / "recipe"
    assert (rec / "serve.sh").is_file() and (rec / "overlay" / "sitecustomize.py").is_file() and (rec / "tools" / "vendor" / "NOTICE").is_file()
    assert not (rec / ".git").exists() and not (rec / ".env").exists() and not (rec / "models").exists()
    assert os.access(rec / "serve.sh", os.X_OK) and os.access(rec / "prepare_pack.py", os.X_OK)
    readme = (pack / "README.md").read_text()
    assert readme.startswith("---\nlicense: mit\n") and "hf download essefree/test-pack" in readme and "not in this repo" in readme
    assert "hf upload-large-folder essefree/test-pack" in r.stdout and '"model-00047-of-00048.safetensors"' in r.stdout
    r2 = sh(sys.executable, ROOT / "tools" / "make_hf_pack.py", pack, "--repo", "essefree/test-pack")
    assert "README.md kept" in r2.stdout and "model-00047" not in r2.stdout.split("--exclude")[1]
    # the recipe copy resolves the pack as its parent
    r = sh(sys.executable, rec / "tools" / "resolve_model.py", "--env", none_env, "--shell")
    assert "MODEL_KIND='pack'" in r.stdout and f"PACK='{pack}'" in r.stdout and "MODEL_SOURCE='inpack'" in r.stdout, r.stdout
    assert f"DIFFBOT='{rec / 'diffbot'}'" in r.stdout, r.stdout

    # a repo uploaded without the Engram shards, served from inside: shards linked back from the diffbot folder
    for s in ENGRAM:
        (pack / s).unlink()
    installed = diffbot / "recipe" / "plans" / "plan-2.9bpw-plan-1017-agent30-c512.json"
    installed.unlink()
    r = sh(sys.executable, rec / "prepare_pack.py", "--env", none_env, "--diffbot", diffbot, "--no-docker", "--skip-download", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "nothing to convert" in r.stdout and "converting" not in r.stdout, r.stdout
    assert all((pack / s).is_file() for s in ENGRAM) and installed.is_file()
    info = json.loads((pack / "PACK_INFO.json").read_text())
    assert "converted_shards" in info, "the original PACK_INFO.json is kept"

    # nobody has the Engram shards and downloads are off: a clear refusal
    for s in ENGRAM:
        (pack / s).unlink()
        (diffbot / s).rename(work / ("hidden-" + s))
    r = sh(sys.executable, rec / "prepare_pack.py", "--env", none_env, "--diffbot", diffbot, "--no-docker", "--skip-download", check=False)
    assert r.returncode != 0 and "Engram shards" in (r.stdout + r.stderr), r.stdout + r.stderr
    # ... via ENGRAM_DIR they are found
    eng = work / "engram-elsewhere"
    eng.mkdir()
    for s in ENGRAM:
        (work / ("hidden-" + s)).rename(eng / s)
    r = sh(sys.executable, rec / "prepare_pack.py", "--env", none_env, "--diffbot", diffbot, "--engram", eng, "--no-docker", "--skip-download", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert all(os.path.samefile(pack / s, eng / s) for s in ENGRAM), "linked from ENGRAM_DIR"

    # from this checkout, pointing --model at the pack
    r = sh(sys.executable, ROOT / "prepare_pack.py", "--env", none_env, "--model", pack, "--diffbot", diffbot, "--no-docker",
           "--skip-download", check=False)
    assert r.returncode == 0 and "nothing to convert" in r.stdout, r.stdout + r.stderr
    # diffbot's recipe missing and downloads off: refused before anything else
    r = sh(sys.executable, ROOT / "prepare_pack.py", "--env", none_env, "--model", pack, "--diffbot", work / "no-diffbot", "--no-docker",
           "--skip-download", check=False)
    assert r.returncode != 0 and "diffbot's recipe/" in (r.stdout + r.stderr), r.stdout + r.stderr
    # PACK_INFO.json left out of a repo: still a pack, and one gets written
    (pack / "PACK_INFO.json").unlink()
    r = sh(sys.executable, ROOT / "prepare_pack.py", "--env", none_env, "--model", pack, "--diffbot", diffbot, "--no-docker",
           "--skip-download", check=False)
    assert r.returncode == 0 and json.loads((pack / "PACK_INFO.json").read_text()).get("assembled_elsewhere"), r.stdout + r.stderr

    print(f"ok: in-pack recipe at {rec} serves {pack} without conversion; Engram completion, plan install, model card, upload hint")
    shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
