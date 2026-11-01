#!/usr/bin/env python3
"""The model card (README.md with YAML front matter) of a served pack hosted on Hugging Face.

Facts (shard count, dense EXL3 K distribution, expert K, size) are read from the pack; the quick start is the
in-pack one (<pack>/recipe/serve.sh). The text is a starting point: edit the result.

usage: pack_readme.py <pack> --repo ORG/NAME [--status TEXT] [--no-engram] [-o <pack>/README.md]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

DIFFBOT_REPO = "diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000"
BASE_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
RECIPE_REPO = "https://github.com/essefree/DeepSeek-v4.1-Flash-EXL3-2.9bpw-diffbot-stack"
FIRST_RECIPE = "https://github.com/essefree/DeepSeek-v4.1-Flash-EXL3-2x-RTX6000Pro"
ENGRAM_SHARDS = ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")
DEFAULT_STATUS = ("**Status: work in progress.** The recipe is being brought up on the author's machine and is not yet "
                  "verified end to end; speed and quality numbers will follow.")


def facts(pack: Path, with_engram: bool) -> dict:
    cfg = json.loads((pack / "config.json").read_text())
    q = cfg.get("quantization_config") or {}
    dense = (q.get("non_routed_exl3") or {}).get("layers") or {}
    kdist = dict(sorted(collections.Counter(int(v["bits"]) for v in dense.values()).items()))
    layer_bits = {int(k): int(v) for k, v in (q.get("layer_bits") or {}).items()}
    idx = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(idx.values()))
    counted = [s for s in shards if with_engram or s not in ENGRAM_SHARDS]
    size = sum(os.path.getsize(pack / s) for s in counted if (pack / s).is_file())
    engram_size = sum(os.path.getsize(pack / s) for s in ENGRAM_SHARDS if (pack / s).is_file())
    return {"dense": len(dense), "kdist": kdist, "layer_bits": layer_bits, "shards": len(shards), "tensors": len(idx),
            "size_gib": size / 2**30, "engram_gib": engram_size / 2**30 if engram_size > 2**30 else 103.0,
            "base_bits": int(q.get("bits", 3)), "codebook": q.get("codebook", "mul1")}


def render(repo: str, pack: Path, status: str, with_engram: bool) -> str:
    f = facts(pack, with_engram)
    name = repo.split("/", 1)[1]
    k2 = sorted(L for L, b in f["layer_bits"].items() if b == 2)
    k2_txt = f"K=2 on layers {k2[0]}–{k2[-1]}" if k2 else "no K=2 layers"
    kdist_txt = ", ".join(f"K={k}: {n}" for k, n in f["kdist"].items())
    disk = f["size_gib"] + (0 if with_engram else f["engram_gib"])
    if with_engram:
        engram_step = (f"the int4 Engram shards `model-00047/48-of-00048.safetensors` ({f['engram_gib']:.0f} GiB) are part of "
                       "this repo: nothing more to fetch;")
        engram_row = "| `model-00047..48-of-00048.safetensors` | diffbot's int4 Engram tables (`int4_fp16s32`), as in their pack |"
    else:
        engram_step = (f"fetches diffbot's int4 Engram shards `model-00047/48-of-00048.safetensors` ({f['engram_gib']:.0f} GiB, "
                       "not part of this repo) from their repo into `recipe/diffbot/`, or links them from `ENGRAM_DIR` if you "
                       "already have them;")
        engram_row = (f"| `model-00047..48-of-00048.safetensors` | **not in this repo**: diffbot's int4 Engram tables, fetched from "
                      f"[{DIFFBOT_REPO}](https://huggingface.co/{DIFFBOT_REPO}) on the first run |")
    return f"""---
license: mit
base_model: {BASE_REPO}
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
tags:
- exl3
- deepseek
- vllm
- rtx-pro-6000
- sm120
---

# DeepSeek-V4.1-Flash-1017 · EXL3 2.9 bpw · 2× RTX PRO 6000

{status}

An EXL3 2.9 bpw quant of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/{BASE_REPO}) (routed experts
K={f['base_bits']} {f['codebook']}, {k2_txt}; EXL3 attention, shared experts and lm_head), packed so that the serving stack of
[{DIFFBOT_REPO}](https://huggingface.co/{DIFFBOT_REPO}) loads it as it is: their vLLM image
(nightly + 1017 patches), their patched `vllm-exl3` plugin with host-expert plans and the VRAM expert cache, their `xmoe`
sm_120 MoE kernels (plus the K=2 instances this quant needs), the int8 decode-once prefill kernel, the source-precision
DSpark drafter (stock 1017 shards 44–46) and their int4 Engram tables. The `recipe/` folder in this repo serves it with
one command; it is the [{RECIPE_REPO.rsplit('/', 1)[1]}]({RECIPE_REPO}) recipe, copied in.

## Hardware

Two sm_120 GPUs with 96 GB each (RTX PRO 6000 Blackwell), a CUDA 13 driver, Docker with the NVIDIA container toolkit,
about 260 GiB of free host RAM (the host experts and the Engram tables are pinned there) and roughly {disk:.0f} GB of disk
for the weights. An x86_64 host.

## Quick start

```bash
pip install -U "huggingface_hub[hf_transfer]"
hf download {repo} --local-dir ~/models/{name} --max-workers 8
cd ~/models/{name}/recipe
./serve.sh
```

The first run, in order (every step is resumable and skipped once done):

1. fetches diffbot's `recipe/` (a few MB: their image build, patches, kernel sources, plans) into `recipe/diffbot/`,
   unless `DIFFBOT=` in `recipe/.env` points at a download of their pack you already have;
2. builds their serving image `dsv41-flash-1017-exl3-3bpw-sm120` with their `docker/build.sh` (long: it compiles
   ExLlamaV3 for sm_120a; skipped when the image exists);
3. {engram_step}
4. builds the K=2 `xmoe` kernel instances in their image (about four minutes, no GPU needed);
5. serves an OpenAI-compatible API on `:8000`, model name `deepseek-v4.1-flash-2.9bpw`, with no authentication
   unless `API_KEY` is set (then clients send it as the Bearer token).

Later runs go straight to step 5.

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \\
  -d '{{"model":"deepseek-v4.1-flash-2.9bpw","messages":[{{"role":"user","content":"hello"}}],"max_tokens":64}}'
```

### Settings

Everything has a default. `cp env.example .env` inside `recipe/` to change `API_KEY`, `PORT`, `SERVED`, `DIFFBOT`
(an existing download of diffbot's pack), `ENGRAM_DIR` (a copy of their Engram shards you already have), and
diffbot's own knobs (`MAX_MODEL_LEN`, `KV_MEM`, `MAX_NUM_SEQS`, `SPEC_K`, `CPU_EXPERTS`, documented in their
`recipe/serve/serve.sh`). From a clone of the recipe instead of this folder: `./serve.sh --model {repo}`
fetches this repo into `models/` and serves it the same way.

## What is in the pack

{f['shards']} shards, {f['tensors']} tensors. The quant's own shards are served unchanged wherever the stack takes EXL3; the
few modules it cannot take as EXL3 were converted once (exact reconstruction with ExLlamaV3, then DeepSeek's own
formats), so the pack is two formats only, like diffbot's: EXL3 for the plugin, DeepSeek block-FP8/BF16 for the rest.

| Files | Content |
|---|---|
| `model-00001..43-of-00043.safetensors` | the quant: routed experts EXL3 K={f['base_bits']} {f['codebook']} ({k2_txt}); {f['dense']} dense EXL3 linears ({kdist_txt}); `attn.wo_a` as DeepSeek block-FP8 (reconstructed from 8 EXL3 slices), `attn.wo_b` / shared `w2` of layers 10–36 as block-FP8 (the quantizer kept them BF16), the DSA indexer `wk` and the KV compressor as BF16, lm_head as BF16 `head.weight` |
| `model-00044..46-of-00048.safetensors` | the stock 1017 DSpark drafter (source precision, `mtp_experts: "source"`) |
{engram_row}
| `config.json` | DeepSeek's config with `quantization_config` in the `vllm-exl3` vocabulary (`layer_bits`, `non_routed_exl3.layers` keyed by vLLM module prefix, `non_routed_quantization` = DeepSeek block-FP8) |
| `plan.json` | the host-expert plan: diffbot's `plan-1017-agent30-c512` with the K=2 layers and the drafter kept in VRAM |
| `recipe/` | the serving recipe (`serve.sh`, the conversion tools, the K=2 kernel sources, the vLLM hooks in `overlay/`, tests) |
| `PACK_INFO.json`, `convert_manifest.json` | how the pack was assembled (conversion errors per module are in the manifest) |

## Credits

* [diffbot](https://huggingface.co/{DIFFBOT_REPO}) for the serving stack this pack rides on (vLLM patches, `vllm-exl3`
  host-expert work, `xmoe` kernels, the int4 Engram tables, the host-expert plans) and for measuring what matters.
* [sfxnz](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark) and [MiaAI-Lab](https://huggingface.co/Mia-AiLab)
  for the DGX Spark EXL3 recipes the quant layout and the first recipe ([{FIRST_RECIPE.rsplit('/', 1)[1]}]({FIRST_RECIPE})) come from.
* [turboderp](https://github.com/turboderp-org/exllamav3) for ExLlamaV3 and the EXL3 format.
* DeepSeek for the model.

## Licenses

Model weights: MIT (DeepSeek). The recipe scripts: MIT. They drive diffbot's recipe (MIT) and its `vllm-exl3` patch
(AGPL-3.0); the `vllm-exl3` pack tools vendored under `recipe/tools/vendor/` are AGPL-3.0 (see `NOTICE` there).
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--repo", required=True, help="Hugging Face repo id ORG/NAME")
    ap.add_argument("--status", default=DEFAULT_STATUS)
    ap.add_argument("--no-engram", action="store_true", help="the Engram shards will not be uploaded")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    pack = Path(a.pack).resolve()
    text = render(a.repo, pack, a.status, not a.no_engram)
    out = Path(a.out) if a.out else pack / "README.md"
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
