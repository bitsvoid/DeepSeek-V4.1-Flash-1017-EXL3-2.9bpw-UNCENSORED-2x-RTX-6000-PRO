# dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED on diffbot's 2× RTX PRO 6000 stack

hf download dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED --local-dir ~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED
cd ~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED && git clone https://github.com/essefree/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED-2x-RTX-6000-PRO.git && && ./recipe/serve.sh

Serve a DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED EXL3 2.9 bpw quant of DeepSeek-V4.1-Flash-**1017** (routed experts K=3 mul1,
K=2 on layers 18–22; EXL3 attention, shared experts, lm_head; stock int4 Engram) through the serving
stack of [diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000):
their vLLM image (nightly-5555a2d8 + 1017 patches), the patched `vllm-exl3` plugin with host-expert
plans and the VRAM cache, the `xmoe` sm_120 MoE kernels, the int8 decode-once prefill kernel, the 8-bit
prefill all-reduce, and optionally their CPU host-expert runtime.

> **Status: designed and assembled offline, not yet booted.** Everything below that touches a GPU
> (the FP8 conversion of a few modules, the K=2 kernel build, the boot itself) runs on the serving box.

This is the second recipe for this quant. The first, [DeepSeek-v4.1-Flash-EXL3-2x-RTX6000Pro](https://github.com/essefree/DeepSeek-v4.1-Flash-EXL3-2x-RTX6000Pro),
ports Mia's 2× DGX Spark recipe to the two cards and serves the same files; it works but decodes at
about half the speed of diffbot's pack on the same hardware.

## Why the gap, and what this recipe keeps from diffbot

The Mia port keeps seven whole MoE layers in host RAM and reads every active expert of those layers over
PCIe on every decode step, with the generic ExLlamaV3 `exl3_moe` kernel, and speculates with the quant's
own 4-bit EXL3 DSpark drafter. diffbot's stack instead:

| | Mia port (first recipe) | diffbot stack (this recipe) |
|---|---|---|
| Experts not in VRAM | layers 33–39 entirely (16.6 GiB/rank), read over PCIe every step | the globally coldest ~30% of experts (a plan from routing counts), plus a 512-slot VRAM cache per GPU that serves 50–80% of the cold routes once warm |
| Decode MoE kernel | ExLlamaV3 `exl3_moe` | `xmoe` flat decode kernel for K=3 mul1 (their own pack's format, so the best-tuned path) |
| Prefill MoE kernel | ExLlamaV3 | decode-once int8 tensor-core kernel (`xmoe_do_i8`) |
| DSpark drafter | the quant's K=4 EXL3 experts | **source-precision 1017 drafter** (stock shards 44–46, FP8/FP4); sfxnz measured Mia's K=4 drafter at 1.33 accepted tokens/step vs 2.77 for the source drafter |
| Draft length | 3 | 5 |
| Cross-GPU all-reduce | NCCL | FlashInfer PCIe-IPC in the decode graphs, 8-bit all-reduce for prefill |
| Dense (attention etc.) | ExLlamaV3 `exl3_gemm` | the same `exl3_gemm` (EXL3 dense linears are not in diffbot's pack; their FP8 dense fast paths do not apply) |

So this recipe should land near diffbot's speed but not at it: the dense projections stay on
`exl3_gemm`, and the K=2 layers use kernel instances diffbot never shipped (added here). Nothing has
been measured yet.

## What is converted, and why

The `vllm-exl3` plugin reads an ExLlamaV3-native pack when `config.json` is rewritten into its
vocabulary (`layer_bits`, `non_routed_exl3.layers`), which its own `tools/exl3_pack_tools` do. Checked
against this quant's shard headers (every tensor, no weights): experts K=3 with layers 18–22 at K=2,
1,766 dense EXL3 linears, codebook mul1, lm_head K=6. What the stack cannot take as it is:

| Module | In the quant | Problem | What `prepare_pack.py` does |
|---|---|---|---|
| `attn.wo_a` (40 layers) | 8 EXL3 K=5 slices per layer (`wo_a.slice.0..7`) | nightly vLLM builds `wo_a` as one grouped (`is_bmm`) linear; the plugin has no path for packed slices | reconstructs the slices with ExLlamaV3 (exact w.r.t. the quantized values), concatenates them into DeepSeek's `[8192, 4096]` layout and stores them in DeepSeek's **block-FP8** format (e4m3 + ue8m0 scales per 32×32), which the stack's FP8 delegate serves |
| `attn.indexer.wk` (4 layers) | EXL3 K=8 | vLLM builds this linear with `quant_config=None` (and the plugin accepts dense K 2–6 only) | reconstruct → BF16 `.weight` (4 tiny 128×512 tensors, as stock) |
| `attn.compressor.wkv`, `.wgate` (layers 2, 8, 14, 20; no gate in layer 20) | EXL3 K=5 | vLLM builds the fused compressor linear with `quant_config=None`: no quant method sees it, the loader wants a BF16 weight | reconstruct → BF16 `.weight [512, 5120]` (as stock) |
| `attn.wo_b`, `ffn.shared_experts.w2` (layers 10–36) | plain BF16 `.weight` (54 tensors) | the FP8 delegate expects scales; stock has these in block-FP8 | BF16 → block-FP8 (lossy at FP8 precision, the same precision DeepSeek ships these in) |
| `head` (lm_head) | EXL3 K=6 | this vLLM's DeepSeek loader routes only the name `head.weight` to lm_head | reconstructed (exact for K=6) to DeepSeek's BF16 `head.weight [129280, 5120]`; diffbot's `DS41_FP8_LMHEAD` can then apply |
| `mtp.*` (drafter) | EXL3 K=4 | no K=4 `xmoe` kernels, and a weak drafter | dropped; stock 1017 shards 44–46 take their place, `mtp_experts: "source"` |
| `layers.{1,14}.engram.wkv` (+ `q/k_weight`) | EXL3 K=5/4 | diffbot's Engram shards already carry DeepSeek's FP8 `wkv` and the BF16 `q/k` next to the int4 tables; two copies would collide | the quant's copies are dropped, the stock ones serve |
| Engram tables | stock fp8 shards 47/48 | too big to pin (diffbot pins Engram in host RAM) | diffbot's int4 shards 47/48, `engram_embed_format: int4_fp16s32` |

Everything else is hardlinked unchanged. The served pack is therefore two formats only, exactly like
diffbot's: EXL3 for the plugin, DeepSeek block-FP8 for the rest.

K=2 mul1 experts (layers 18–22) get the missing `xmoe` instances (`exl3_moe_inst_k2_cb2.cu` and the
`dof_gemm` dispatch) in `kernels/`; the template kernels already support them. Those layers stay fully in
VRAM (the cold-path prefill kernel exists for K=3 only), which the plan enforces.

## Layout

| Path | Role |
|---|---|
| `prepare_pack.py` | assembles the served pack from the quant, the stock drafter shards and diffbot's Engram shards; runs the FP8 conversion in diffbot's image; rewrites `config.json`; regenerates the index |
| `tools/convert_natives.py` | GPU step (inside diffbot's image): EXL3 → BF16 via ExLlamaV3, BF16 → DeepSeek block-FP8 |
| `tools/make_plan.py` | diffbot's host-expert plan with the K=2 layers (and the drafter) kept in VRAM |
| `tools/vendor/` | `vllm-exl3` pack tools at commit `d3cfd394` (AGPL-3.0, see `NOTICE`) |
| `kernels/add_k2_mul1.py`, `kernels/build.sh` | K=2 mul1 instances for `xmoe` / `xmoe_do_i8`, built with diffbot's scripts in their image |
| `serve.sh` | runs diffbot's `recipe/serve/serve.sh` with this pack, plan, kernels and the knobs that must differ; fetches/builds what is missing on the first run |
| `tools/resolve_model.py` | `MODEL=<dir \| org/name>`, raw quant vs served pack, the in-pack default (`<pack>/recipe`) |
| `tools/make_hf_pack.py`, `tools/pack_readme.py` | package a served pack for Hugging Face: `recipe/` inside it + the model card; prints the upload command |
| `overlay/sitecustomize.py` | runs diffbot's sitecustomize, then two hooks this quant needs: vLLM's block-FP8 shared-expert padding skips EXL3 tensors, and vllm-exl3 loads EXL3 modules on a `ReplicatedLinear` (the indexer `wq_b`) unsharded (its TP geometry comes from `LinearBase`, which carries the global TP size even for replicated layers) |
| `tests/` | offline checks against a header-only mock of the quant (`fetch_headers.py` rebuilds the fixture shards from the public repos) |

## Quick start (on the box)

Prerequisites: two sm_120 GPUs with 96 GB, a CUDA 13 driver, Docker with the NVIDIA container toolkit, ~260 GiB of
free host RAM, the Hugging Face CLI (`pip install -U "huggingface_hub[hf_transfer]"`). Everything else is fetched or
built on the first run: diffbot's `recipe/` (into `./diffbot`, unless `DIFFBOT=` in `.env` points at a download of
their pack you have), their serving image (their `docker/build.sh`; long, it compiles ExLlamaV3 for sm_120a), their
int4 Engram shards 47/48 (~103 GiB, unless the pack carries them or `ENGRAM_DIR` has a copy, e.g. the first recipe's
`engram-int4`), the K=2 kernels.

**From the served pack on Hugging Face** (the recipe is inside it as `recipe/`; nothing to convert):

```bash
hf download <org>/<name> --local-dir ~/models/<name> --max-workers 8
cd ~/models/<name>/recipe
./serve.sh                                              # .env optional (cp env.example .env to change knobs)
```

**From this checkout**, pointing at the raw quant or at a served pack, local or on Hugging Face:

```bash
git clone https://github.com/essefree/DeepSeek-v4.1-Flash-EXL3-2.9bpw-diffbot-stack.git && cd DeepSeek-v4.1-Flash-EXL3-2.9bpw-diffbot-stack
./serve.sh --model /data/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw     # the raw quant: converted + assembled first
./serve.sh --model <org>/<name>                                           # a Hugging Face repo (fetched into ./models/)
```

The API has no authentication unless `API_KEY=…` is set (then clients send it as the Bearer token).
diffbot's `serve.sh` requires a key, so the wrapper runs a copy of it with the key made optional
(`tools/patch_diffbot_serve.py`, regenerated on every start).

`--model` (or `MODEL=` in `.env`) is a local directory or a Hugging Face repo id, and either the raw quant or a
served pack; `tools/resolve_model.py` tells them apart (`PACK_INFO.json`, or a `config.json` already in the plugin's
vocabulary). With the raw quant the first run fetches the stock 1017 drafter (7.4 GiB), converts the few modules in
diffbot's image (GPU, ~30 min once) and assembles the pack as `<quant>-served` next to the quant (hardlinks, so ~25 GB
of new data). With a served pack nothing is converted: missing Engram shards are completed and the plan installed.
Then the K=2 kernels are built into diffbot's `recipe/kernels/build/` and it serves on `:8000`. Later runs go
straight to serving. The steps also run on their own: `./prepare_pack.py --model ...`, `bash kernels/build.sh`;
`NO_AUTO_PREPARE=1 ./serve.sh` only checks.

## Hosting the pack on Hugging Face

The served pack is what gets hosted (the raw quant would make every downloader convert), with this recipe copied in
as `recipe/`, the way diffbot's pack ships theirs:

```bash
./prepare_pack.py --model /data/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw          # once; -> <quant>-served
python3 tools/make_hf_pack.py /data/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-served --repo <org>/<name> [--no-engram]
```

`make_hf_pack.py` copies the tracked files of this checkout into `<pack>/recipe` (no `.git`, `.env`, caches or
downloads), writes the model card `<pack>/README.md` from `tools/pack_readme.py` (facts read from the pack; edit it,
`--status` sets its status line, `--force-readme` regenerates), and prints the upload command: `hf upload-large-folder`
with exclusions for local state. `--no-engram` leaves diffbot's Engram shards 47/48 (103 GiB of their data) out of
the upload; downloaders then fetch them from diffbot's repo on the first run. `PACK_INFO.json` and
`convert_manifest.json` go up as provenance (they name the paths the pack was assembled from).

## Tests (no GPU)

`tests/test_offline.py` builds a header-only mock of the quant from its index and `quantization_config.json`,
fakes the conversion exactly as `convert_natives.py` would rewrite each shard, mocks diffbot's pack and the
stock drafter from real shard headers, runs `prepare_pack.py` on it and checks the result: 41 shards
rewritten (40 layer shards for `wo_a`, shard 43 for lm_head and the drafter leftovers), 154 dense EXL3 modules keyed by
vLLM prefix (K 4/5/6), drafter and Engram names present, no stale slices.

`tests/test_pack_mode.py` takes that pack the way it is hosted: copies the recipe into it (`make_hf_pack.py`), removes the
Engram shards, and runs `<pack>/recipe/prepare_pack.py` with no `--model`: the pack is found as the parent, the shards
linked back (or refused with `--skip-download` when nobody has them), the plan installed, nothing converted.

```bash
python3 tests/fetch_headers.py diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 <engram dir> model-00047-of-00048.safetensors model-00048-of-00048.safetensors
python3 tests/fetch_headers.py deepseek-ai/DeepSeek-V4.1-Flash-1017 <drafter dir> model-00044-of-00048.safetensors model-00045-of-00048.safetensors model-00046-of-00048.safetensors --revision 1bae0e1088993b9b82de08f59de5be5a34e42170
python3 tests/test_offline.py --quant-meta <dir with the quant's json files> --stock-drafter <drafter dir> --engram <engram dir>
python3 tests/test_pack_mode.py --quant-meta <dir> --stock-drafter <drafter dir> --engram <engram dir>
python3 tests/test_sitecustomize_hook.py   # the vLLM / vllm-exl3 hooks, against fake modules
```

## Open items (box-verified)

1. The FP8 delegate's handling of the converted modules, and `wo_a` through the FP8 grouped path.
2. `fused_wqa_wkv` is matched by prefix in the plugin (box-verified); the compressor is not EXL3 at all
   (vLLM builds it unquantized, see the table); the Engram projection (`engram.wkv`) is DeepSeek's FP8 one
   from diffbot's shards, not the quant's EXL3 copy.
3. The K=2 `xmoe` instances compile and the plugin picks them for layers 18–22.
4. VRAM: estimated ~89 GiB/rank with diffbot's plan minus the K=2 layers; `KV_MEM` is the lever.
5. Speed and KL against the Mia port and against diffbot's pack.

## Licenses

Scripts here: MIT. They drive diffbot's recipe (MIT) and its `vllm-exl3` patch (AGPL-3.0); the vendored
`vllm-exl3` tools are AGPL-3.0. Model weights are MIT (DeepSeek).
