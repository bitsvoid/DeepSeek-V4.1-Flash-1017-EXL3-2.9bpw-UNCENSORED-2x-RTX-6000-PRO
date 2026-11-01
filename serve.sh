#!/usr/bin/env bash
# Serve the pack through diffbot's recipe/serve/serve.sh with the settings this quant needs.
#
# Everything else (image, patches, mounts, API, CPU_EXPERTS, 1M context, DSpark k=5, warmup) is theirs;
# their script documents every knob and all of them can still be set in the environment.
#
#   ./serve.sh                                         # inside a served pack (<pack>/recipe, as hosted on Hugging Face),
#                                                      # or MODEL= from .env
#   ./serve.sh --model essefree/my-exl3-quant          # a Hugging Face repo (fetched into ./models/): a served pack or the quant
#   ./serve.sh --model /data/models/my-exl3-quant      # a directory: a served pack or the quant
# No authentication unless API_KEY is set (API_KEY=secret ./serve.sh -> clients send it as the Bearer
# token); diffbot's script requires a key, so a patched copy of it runs (tools/patch_diffbot_serve.py).
# The first run does the whole path: fetch the model (HF), fetch diffbot's recipe/ if DIFFBOT holds none,
# build their serving image if it is missing (long: ExLlamaV3 for sm_120a), assemble the pack when the model
# is the raw quant (prepare_pack.py: stock drafter download + GPU conversion, ~30 min once; a served pack
# needs none of that), build the K=2 kernels (kernels/build.sh), then serve. NO_AUTO_PREPARE=1 only checks.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL_ARG="$2"; shift 2 ;;
        --model=*) MODEL_ARG="${1#--model=}"; shift ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown argument $1 (only --model <dir | hf-org/name>)"; exit 2 ;;
    esac
done
# .env is optional: inside a served pack every setting has a default (env.example lists them)
[ -f "$ROOT/.env" ] && { set -a; source "$ROOT/.env"; set +a; }
export API_KEY=${API_KEY:-}
model_args=()
[ -n "$MODEL_ARG" ] && model_args=(--model "$MODEL_ARG")
resolve() { python3 "$ROOT/tools/resolve_model.py" --env "$ROOT/.env" ${model_args[@]+"${model_args[@]}"} --shell; }
resolved="$(resolve)" || exit 2
eval "$resolved"
export DIFFBOT
REC=$DIFFBOT/recipe
export IMAGE=${IMAGE:-dsv41-flash-1017-exl3-3bpw-sm120}
PLAN_NAME=${PLAN_NAME:-plan-2.9bpw-${DIFFBOT_PLAN:-plan-1017-agent30-c512.json}}
have_image() { docker image inspect "$IMAGE" >/dev/null 2>&1; }
have_kernels() { [ -d "$REC/kernels/build/xmoe_k2" ] && [ -d "$REC/kernels/build/k2/xmoe_do_i8" ]; }
if [ "${NO_AUTO_PREPARE:-0}" = "1" ]; then
    [ -f "$REC/serve/serve.sh" ] || { echo "diffbot's recipe missing under $DIFFBOT (./prepare_pack.py fetches it)"; exit 1; }
    [ -f "$PACK/PACK_INFO.json" ] && [ -f "$REC/plans/$PLAN_NAME" ] || { echo "pack $PACK not assembled: ./prepare_pack.py --model '$MODEL'"; exit 1; }
    have_image || { echo "image $IMAGE missing: TAG=$IMAGE bash $REC/docker/build.sh"; exit 1; }
    have_kernels || { echo "K=2 kernels missing: bash kernels/build.sh"; exit 1; }
else
    # Always: the pack must reflect the current recipe (a converter fix re-converts only the shards it
    # concerns; the rest is header scans and a config / index / plan rebuild, seconds on a finished pack;
    # a served pack is only completed and checked). This also fetches the model and diffbot's recipe/.
    echo "[serve] prepare_pack.py --model '$MODEL'"
    python3 "$ROOT/prepare_pack.py" --env "$ROOT/.env" --model "$MODEL"
    resolved="$(resolve)" || exit 2      # a fetched repo may have turned out to be a served pack: PACK moves
    eval "$resolved"
    if ! have_image; then
        echo "[serve] serving image $IMAGE not built yet -> TAG=$IMAGE bash $REC/docker/build.sh (long: compiles ExLlamaV3 for sm_120a)"
        TAG=$IMAGE bash "$REC/docker/build.sh"
        have_image || { echo "[serve] ERROR: $IMAGE still missing after docker/build.sh" >&2; exit 1; }
    fi
    if ! have_kernels; then
        echo "[serve] K=2 mul1 kernels not built yet -> kernels/build.sh"
        bash "$ROOT/kernels/build.sh"
    fi
fi
export PACK

# The pack's parent is what their serve.sh mounts (MODELS_DIR); the drafter is inside the pack.
export MODELS_DIR=${MODELS_DIR:-$(dirname "$PACK")}
# Engram shards linked from outside MODELS_DIR (ENGRAM_DIR on another filesystem = symlinks) need their own mount.
extra=""
for s in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
    [ -L "$PACK/$s" ] || continue
    d=$(dirname "$(readlink -f "$PACK/$s")")
    case "$d" in "$MODELS_DIR"|"$MODELS_DIR"/*) ;; *) [[ "$extra" == *"-v $d:$d:ro"* ]] || extra="$extra -v $d:$d:ro" ;; esac
done
export DOCKER_EXTRA="${DOCKER_EXTRA:-}$extra"
export SERVED=${SERVED:-deepseek-v4.1-flash-2.9bpw} PORT=${PORT:-8000} NAME=${NAME:-dsv41-exl3-29}
export PLAN=$PLAN_NAME HOST_CACHE=${HOST_CACHE:-512}
# K=2 mul1 instances for layers 18-22 (kernels/build.sh)
export XMOE_EXT=xmoe_k2 EXL3_DO=1 DO_INT8=1 EXL3_DO_MOD=xmoe_do_i8 EXL3_DO_DIR=/opt/recipe/kernels/build/k2/xmoe_do_i8
# The drafter is the stock 1017 head inside the pack (source format), not a separate folder.
export DS41_DRAFTER=released
# diffbot's FP8 fast paths assume FP8 dense tensors; this quant's attention/shared/head are EXL3.
# They check the dtype and skip, but keep them off so the log is unambiguous. wo_a, wo_b and the
# shared down-proj ARE block-FP8 here (convert_natives.py), so DS41_WOA_FP8 / DENSE_GEMV could apply
# to those; measure before turning them on.
export DS41_DENSE_GEMV=${DS41_DENSE_GEMV:-0} DS41_DENSE_ONE_COPY=${DS41_DENSE_ONE_COPY:-0}
export DS41_WOA_FP8=${DS41_WOA_FP8:-0} DS41_FP8_LMHEAD=${DS41_FP8_LMHEAD:-0}
export LINEAR_BACKEND=${LINEAR_BACKEND-}          # no Marlin MXFP8 dense repack (nothing to repack)
export DENSE_HYBRID_M=${DENSE_HYBRID_M:-0}
# diffbot's serve.sh with API_KEY optional, SERVED honoured and this recipe's sitecustomize mounted.
# Regenerated on every start (cheap, and a stale copy once mounted diffbot's sitecustomize after a pull).
DSERVE=$ROOT/.cache/diffbot-serve.sh
python3 "$ROOT/tools/patch_diffbot_serve.py" "$REC" "$DSERVE"
if ! grep -q 'overlay/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py' "$DSERVE"; then
    echo "[serve] ERROR: $DSERVE does not mount $ROOT/overlay/sitecustomize.py" >&2; exit 1
fi
echo "[serve] pack $PACK ($MODEL_KIND; 2.9 bpw EXL3 on diffbot's stack), plan $PLAN, kernels $XMOE_EXT + $EXL3_DO_DIR, auth $([ -n "$API_KEY" ] && echo 'Bearer token' || echo none)${extra:+, extra mounts:$extra}"
exec bash "$DSERVE"
