#!/usr/bin/env python3
"""Assemble the served pack: the 2.9 bpw quant in the form diffbot's stack loads.

  PACK/
    model-000NN-of-00043.safetensors   the quant's shards: hardlinked, except the ones that held wo_a
                                       slices / indexer.wk, compressor wkv+wgate / BF16 wo_b, w2 / the Engram projection,
                                       which tools/convert_natives.py rewrote (DeepSeek block-FP8, BF16)
    model-00044..46-of-00048.safetensors  stock DeepSeek-V4.1-Flash-1017 drafter (hardlinked)
    model-00047..48-of-00048.safetensors  diffbot's int4 Engram tables (hardlinked from ENGRAM_DIR, default
                                       the diffbot pack; any folder holding those two files works, e.g. the
                                       first recipe's ./engram-int4; fetched from diffbot's repo when nobody has them)
    config.json                        plugin vocabulary (tools/vendor + tools/rewrite_config.py)
    model.safetensors.index.json       rebuilt from the shard headers
    plan.json                          host-expert plan (also installed into diffbot's recipe/plans/)
    PACK_INFO.json                     what went in

Steps that need a GPU (the conversion) run in diffbot's image through docker. Everything else is
header-only Python on the host. Re-runnable: finished steps are skipped.

Given a served pack instead of the quant (a download of one from Hugging Face, or this recipe run from
<pack>/recipe), there is nothing to convert: the Engram shards are completed if the repo left them out,
the plan is installed into diffbot's recipe (fetched when DIFFBOT holds nothing yet) and the pack is checked.

usage: prepare_pack.py [--model <dir | hf-org/name>] [--env .env] [--skip-download] [--skip-convert]
                       [--no-docker] [--force-links]
The model comes from --model / MODEL in .env (a directory, or a Hugging Face repo fetched into
<recipe>/models/), or is the parent directory when the recipe sits inside a served pack; a quant's pack
goes to PACK or <quant>-served.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
VENDOR = TOOLS / "vendor"
sys.path.insert(0, str(TOOLS))
import convert_natives  # noqa: E402  (plan_shards, header-only)
import resolve_model  # noqa: E402  (MODEL=<dir | hf-org/name>, PACK default, served-pack detection)

STOCK_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
STOCK_REV = "1bae0e1088993b9b82de08f59de5be5a34e42170"
DIFFBOT_REPO = "diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000"
DRAFTER_SHARDS = tuple(f"model-000{n}-of-00048.safetensors" for n in (44, 45, 46))
ENGRAM_SHARDS = ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")
COPY_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def read_env(path: Path) -> dict[str, str]:
    return resolve_model.read_env(path)


def header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def link(src: Path, dst: Path, force: bool) -> str:
    if dst.exists() or dst.is_symlink():
        if not force:
            try:
                if dst.stat().st_ino == src.stat().st_ino and dst.stat().st_dev == src.stat().st_dev:
                    return "exists"
            except OSError:
                pass
            if dst.is_symlink() and os.path.realpath(dst) == os.path.realpath(src):
                return "exists"
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        os.symlink(os.path.realpath(src), dst)
        return "symlink"


def run(cmd: list[str], **kw) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def hf_cli() -> str:
    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    if not hf:
        raise SystemExit("need the Hugging Face CLI: pip install -U 'huggingface_hub[hf_transfer]'")
    return hf


def hf_download(repo: str, dest: Path, includes: tuple[str, ...] = (), revision: str = "") -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [hf_cli(), "download", repo, "--local-dir", dest, "--max-workers", os.environ.get("HF_MAX_WORKERS", "8")]
    if revision:
        cmd += ["--revision", revision]
    for inc in includes:
        cmd += ["--include", inc]
    run(cmd, env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1")})


def engram_format(path: Path) -> str:
    h = header(path)
    for k, v in h.items():
        if k.endswith(".engram.embed.weight"):
            return "int4" if v["dtype"] == "U8" and v["shape"][1] == 128 else "fp8"
    return "none"


def ensure_diffbot_recipe(diffbot: Path, plan_name: str, allow_download: bool, revision: str) -> Path:
    """diffbot's recipe/ (image build, serve.sh, patches, kernel sources, plans) under DIFFBOT; fetched from
    their repo when missing (recipe/ only, a few MB: the pack itself is not needed)."""
    rec = diffbot / "recipe"
    if not (rec / "serve" / "serve.sh").is_file():
        if not allow_download:
            raise SystemExit(f"{rec}: diffbot's recipe/ is not there (--skip-download given); set DIFFBOT to a download "
                             f"of {DIFFBOT_REPO}, or let this script fetch recipe/ into {diffbot}")
        log(f"fetching diffbot's recipe/ ({DIFFBOT_REPO}{'@' + revision if revision else ''}) -> {diffbot}")
        hf_download(DIFFBOT_REPO, diffbot, includes=("recipe/*",), revision=revision)
        if not (rec / "serve" / "serve.sh").is_file():
            raise SystemExit(f"{rec}/serve/serve.sh still missing after the download")
    if not (rec / "plans" / plan_name).is_file():
        raise SystemExit(f"{rec}/plans/{plan_name} missing (DIFFBOT_PLAN names one of diffbot's recipe/plans/*.json)")
    return rec


def ensure_engram(pack: Path, engram_dir: Path | None, diffbot: Path, allow_download: bool, force_links: bool) -> Path:
    """diffbot's two int4 Engram shards in the pack: already there, else linked from ENGRAM_DIR or the diffbot
    folder, else fetched (~103 GiB) into ENGRAM_DIR (or the diffbot folder) and linked. Returns where they came from."""
    def holds(d: Path) -> bool:
        return all((d / s).is_file() and os.access(d / s, os.R_OK) for s in ENGRAM_SHARDS)

    def check_int4(d: Path) -> None:
        for s in ENGRAM_SHARDS:
            if engram_format(d / s) != "int4":
                raise SystemExit(f"{d / s} does not hold int4 Engram tables (stock fp8? diffbot's are U8 [N,128] + F16 [N,8])")

    if holds(pack) and not any((pack / s).is_symlink() and not (pack / s).exists() for s in ENGRAM_SHARDS):
        check_int4(pack)
        return pack
    for cand in (engram_dir, diffbot):
        if cand is not None and holds(cand):
            check_int4(cand)
            for s in ENGRAM_SHARDS:
                link(cand / s, pack / s, force_links)
            return cand
    dest = engram_dir or diffbot
    if not allow_download:
        raise SystemExit(f"diffbot's int4 Engram shards {ENGRAM_SHARDS} are in neither {pack}, "
                         f"{engram_dir or '(ENGRAM_DIR unset)'} nor {diffbot} (--skip-download given)")
    log(f"fetching diffbot's int4 Engram shards 47/48 (~103 GiB, resumable) from {DIFFBOT_REPO} -> {dest}")
    hf_download(DIFFBOT_REPO, dest, includes=ENGRAM_SHARDS)
    if not holds(dest):
        raise SystemExit(f"Engram shards still missing under {dest} after the download")
    check_int4(dest)
    for s in ENGRAM_SHARDS:
        link(dest / s, pack / s, force_links)
    return dest


def verify_pack(pack: Path) -> tuple[list[str], dict]:
    """Problems with the assembled pack (empty = fine) and the index weight map."""
    idx = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
    names = set(idx)
    problems = []
    for bad in ("wo_a.slice.", ".indexer.wk.trellis", ".compressor.wkv.trellis", ".compressor.wgate.trellis",
                ".engram.wkv.trellis", "head.trellis"):
        hits = [n for n in names if bad in n]
        if hits:
            problems.append(f"{len(hits)} unconverted tensors like {hits[0]}")
    if not any(n.startswith("mtp.") for n in names):
        problems.append("no mtp.* tensors (drafter missing)")
    if not any(n.endswith(".engram.embed.weight") for n in names):
        problems.append("no Engram tables")
    if "head.weight" not in names:
        problems.append("no head.weight (lm_head)")
    cfg = json.loads((pack / "config.json").read_text()).get("quantization_config") or {}
    for key in ("layer_bits", "non_routed_exl3", "non_routed_quantization", "mtp_experts", "engram_embed_format"):
        if key not in cfg:
            problems.append(f"config.json lacks quantization_config.{key}")
    missing = sorted({f for f in idx.values() if not (pack / f).is_file()})
    if missing:
        problems.append(f"{len(missing)} shards named by the index are missing (e.g. {missing[0]})")
    return problems, idx


def install_plan(rec: Path, plan_name: str, pack: Path) -> Path:
    run([sys.executable, TOOLS / "make_plan.py", rec / "plans" / plan_name, pack / "plan.json"])
    installed = rec / "plans" / f"plan-2.9bpw-{plan_name}"
    shutil.copy2(pack / "plan.json", installed)
    log(f"plan installed as {installed} (serve.sh PLAN={installed.name})")
    return installed


def finish_served_pack(pack: Path, rec: Path, plan_name: str, engram_dir: Path | None, diffbot: Path, a) -> int:
    """A pack that was assembled already (downloaded, or this recipe runs inside it): complete and check it."""
    pack_info = pack / "PACK_INFO.json"
    if not os.access(pack, os.W_OK):
        raise SystemExit(f"{pack} is not writable (the plan and any missing Engram links go there)")
    src = ensure_engram(pack, engram_dir, diffbot, not a.skip_download, a.force_links)
    log(f"Engram: int4 tables {'in the pack' if src == pack else 'from ' + str(src)}")
    installed = install_plan(rec, plan_name, pack)
    problems, idx = verify_pack(pack)
    if not pack_info.is_file():
        pack_info.write_text(json.dumps({"assembled_elsewhere": True, "plan": {"from": plan_name, "installed": str(installed)},
                                         "tensors": len(idx), "shards": len(set(idx.values())),
                                         "checked": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2) + "\n")
    log(f"served pack: {len(set(idx.values()))} shards, {len(idx)} tensors")
    if problems:
        for p in problems:
            log("PROBLEM " + p)
        return 1
    log(f"OK -> {pack} (nothing to convert)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=str(HERE / ".env"))
    ap.add_argument("--model", help="the quant or a served pack: a directory or a Hugging Face repo id (MODEL in .env)")
    ap.add_argument("--quant", help="alias of --model for a directory")
    ap.add_argument("--diffbot"), ap.add_argument("--pack")
    ap.add_argument("--engram", help="directory with diffbot's int4 model-00047/48-of-00048.safetensors (ENGRAM_DIR; default: DIFFBOT)")
    ap.add_argument("--stock-drafter"), ap.add_argument("--image"), ap.add_argument("--diffbot-plan")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-convert", action="store_true", help="converted shards are already in PACK")
    ap.add_argument("--no-docker", action="store_true", help="do not check the image (tests)")
    ap.add_argument("--force-links", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    env = read_env(Path(a.env))
    get = lambda key, flag: Path(flag) if flag else (Path(env[key]) if env.get(key) else None)
    try:
        rm = resolve_model.resolve(a.model or a.quant, env, a.pack)
    except ValueError as exc:
        raise SystemExit(str(exc))
    quant, pack = Path(rm["QUANT"]), Path(rm["PACK"])
    diffbot = get("DIFFBOT", a.diffbot) or Path(rm["DIFFBOT"])
    engram_dir = get("ENGRAM_DIR", a.engram)
    log(f"model: {rm['MODEL']} ({rm['MODEL_SOURCE']}, {rm['MODEL_KIND']}) -> {quant}; pack -> {pack}")
    stock = get("STOCK_DRAFTER", a.stock_drafter) or (HERE / "drafter-1017")
    image = a.image or env.get("IMAGE") or "dsv41-flash-1017-exl3-3bpw-sm120"
    plan_name = a.diffbot_plan or env.get("DIFFBOT_PLAN") or "plan-1017-agent30-c512.json"

    # ---- 1. inputs (an HF repo is fetched whole: the EXL3 shards are the bulk of it)
    ok, why = resolve_model.quant_complete(quant)
    if not ok and rm["MODEL_SOURCE"] == "hf":
        if a.skip_download:
            raise SystemExit(f"{quant}: {why} (--skip-download given)")
        log(f"fetching {rm['HF_MODEL_REPO']} -> {quant} (resumable; private repos need HF_TOKEN)")
        hf_download(rm["HF_MODEL_REPO"], quant)
        rm = resolve_model.resolve(a.model or a.quant, env, a.pack)   # a served pack is only known once fetched
        quant, pack = Path(rm["QUANT"]), Path(rm["PACK"])
        ok, why = resolve_model.quant_complete(quant)
    rec = ensure_diffbot_recipe(diffbot, plan_name, not a.skip_download, os.environ.get("DIFFBOT_REV") or env.get("DIFFBOT_REV", ""))
    if rm["MODEL_KIND"] == "pack":
        # a repo hosted without diffbot's Engram shards is "incomplete" only by those two files
        if not ok and not _only_engram_missing(quant):
            raise SystemExit(f"{quant}: {why}")
        log(f"{quant} is a served pack (PACK_INFO.json / plugin config.json): nothing to convert")
        return finish_served_pack(quant, rec, plan_name, engram_dir, diffbot, a)
    if not ok:
        raise SystemExit(f"{quant}: {why}")
    if not a.no_docker and not a.skip_convert:
        if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
            raise SystemExit(f"docker image {image} missing: build it with {rec}/docker/build.sh")
    pack.mkdir(parents=True, exist_ok=True)
    engram_src = ensure_engram(pack, engram_dir, diffbot, not a.skip_download, a.force_links)
    log(f"Engram: int4 tables from {engram_src}")
    # roles from the shard headers, not the index (the quant's index is stale)
    roles = convert_natives.quant_shards(quant)
    quant_shards = sorted(roles)
    mtp_only = {s for s, r in roles.items() if r == "drafter"}
    engram_named = {s for s, r in roles.items() if r == "engram" or s in ENGRAM_SHARDS}
    log(f"quant: {len(quant_shards)} shard files, {len(mtp_only)} drafter-only (dropped), "
        f"{len(engram_named)} Engram (replaced by diffbot's)")

    # ---- 2. stock drafter
    stock.mkdir(parents=True, exist_ok=True)
    missing = [s for s in DRAFTER_SHARDS if not (stock / s).is_file()]
    if missing:
        if a.skip_download:
            raise SystemExit(f"stock drafter shards missing under {stock}: {missing}")
        hf_download(STOCK_REPO, stock, includes=tuple(missing), revision=STOCK_REV)
    for s in DRAFTER_SHARDS:
        h = header(stock / s)
        if not all(k.startswith("mtp.") for k in h):
            raise SystemExit(f"{stock / s}: not a DeepSeek drafter shard (non-mtp tensors)")

    # ---- 3. conversion plan + GPU conversion
    plan = convert_natives.plan_shards(quant)
    affected = sorted(plan)
    log(f"conversion: {len(affected)} shards hold wo_a slices / indexer.wk, compressor / BF16 natives / Engram projection")
    manifest = pack / "convert_manifest.json"
    # a converted shard counts as done only when its header holds nothing left to convert or drop
    # (an older run may have missed tensors the index did not list)
    done = {s for s in affected if (pack / s).is_file() and not (pack / s).is_symlink()
            and not _same_file(pack / s, quant / s) and os.access(pack / s, os.R_OK)
            and not convert_natives.needs_work(header(pack / s))}
    todo = [s for s in affected if s not in done]
    if todo and a.skip_convert:
        raise SystemExit(f"--skip-convert but {len(todo)} converted shards are missing in {pack}: {todo[:3]}...")
    if todo:
        log(f"converting {len(todo)} shards in {image} on {a.device} (~17 GB of output) ...")
        # as the calling user (like diffbot's kernel builds), so the shards it writes are ours to read
        run(["docker", "run", "--rm", "--gpus", "all", "--ipc", "host",
             "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
             "-e", "VLLM_NO_USAGE_STATS=1", "-e", "DO_NOT_TRACK=1",
             "-v", f"{quant}:/quant:ro", "-v", f"{pack}:/pack", "-v", f"{TOOLS}:/tools:ro",
             "--entrypoint", "python3", image, "/tools/convert_natives.py", "--src", "/quant", "--out", "/pack",
             "--device", a.device, "--only-shards", ",".join(todo), "--manifest", "/pack/convert_manifest.json"])
    unreadable = [s for s in affected if not (pack / s).is_file() or not os.access(pack / s, os.R_OK)]
    if unreadable:
        raise SystemExit(f"{len(unreadable)} converted shards missing or not readable (e.g. {pack / unreadable[0]}). "
                         f"Shards written by an older run belong to root: sudo chown -R {os.getuid()}:{os.getgid()} '{pack}'")

    # ---- 4. links
    linked = 0
    for s in quant_shards:
        if s in plan or s in mtp_only or s in engram_named:
            continue
        link(quant / s, pack / s, a.force_links)
        linked += 1
    for s in DRAFTER_SHARDS:
        link(stock / s, pack / s, a.force_links)
    for f in COPY_FILES:
        if (quant / f).is_file():
            shutil.copy2(quant / f, pack / f)
    log(f"linked {linked} quant shards unchanged, "
        f"{len(DRAFTER_SHARDS)} drafter, {len(ENGRAM_SHARDS)} Engram; copied {', '.join(f for f in COPY_FILES if (quant / f).is_file())}")

    # ---- 5. config (plugin vocabulary) and index
    py = sys.executable
    run([py, "-I", VENDOR / "qwen_pack_scan.py", pack, "--out", pack / "pack_scan.json"], stdout=subprocess.DEVNULL)
    run([py, "-I", VENDOR / "qwen_pack_config.py", pack, "--scan", pack / "pack_scan.json"])
    run([py, TOOLS / "rewrite_config.py", pack])
    run([py, "-I", VENDOR / "regenerate_safetensors_index.py", pack])

    # ---- 6. plan
    installed = install_plan(rec, plan_name, pack)

    # ---- 7. verify
    problems, idx = verify_pack(pack)
    total = sum(os.path.getsize(pack / s) for s in set(idx.values()) if (pack / s).is_file())
    info = {
        "quant": str(quant), "diffbot": str(diffbot), "engram_dir": str(engram_src),
        "stock_drafter": {"repo": STOCK_REPO, "revision": STOCK_REV},
        "converted_shards": affected, "dropped_quant_shards": sorted(mtp_only | engram_named),
        "plan": {"from": plan_name, "installed": str(installed)},
        "tensors": len(idx), "shards": len(set(idx.values())), "bytes": total,
        "convert_manifest": str(manifest) if manifest.is_file() else None,
        "assembled": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (pack / "PACK_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    log(f"pack: {info['shards']} shards, {info['tensors']} tensors, {total / 2**30:.1f} GiB referenced")
    if problems:
        for p in problems:
            log("PROBLEM " + p)
        return 1
    log(f"OK -> {pack}")
    return 0


def _only_engram_missing(pack: Path) -> bool:
    """A served pack hosted without diffbot's Engram shards: every other shard of the index is there."""
    idx = pack / "model.safetensors.index.json"
    if not idx.is_file():
        return False
    wm = json.loads(idx.read_text())["weight_map"]
    missing = {f for f in wm.values() if not (pack / f).is_file()}
    return bool(missing) and missing <= set(ENGRAM_SHARDS)


def _same_file(a: Path, b: Path) -> bool:
    try:
        sa, sb = a.stat(), b.stat()
        return sa.st_ino == sb.st_ino and sa.st_dev == sb.st_dev
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(main())
