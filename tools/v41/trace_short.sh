#!/bin/bash
# Engine half of the reference comparison (pairs with refprefix.py). Run with llama-swap STOPPED.
#   bash trace_short.sh <image> "<prompt text>" [out]
# 1) --dump-tokens: the ids the engine encodes for the text (BOS added), fed to refprefix.py --ids;
# 2) --dump-logits 10 --tokens <ids> with AFF_NORM_TRACE=1: top-10 next-token logits + per-sublayer
#    sums of squares, engram and draft OFF, static placement.
set -uo pipefail
IMG=${1:?image}; TEXT=${2:-The capital of France is}; OUT=${3:-/mnt/models2/affinity/v41/trace_$(date +%H%M)}
M=/mnt/models2/affinity/v41
run() { docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt label=disable --network host \
  --ipc host --ulimit memlock=-1:-1 --shm-size 16g -e HIP_VISIBLE_DEVICES=0,1 "$@"; }
common=(-m /m/model.aff --tokenizer /ck/tokenizer.json --dspark off --host-pool-mib 138000
        --host-pool-reserve-mib 512 --ui plain --kv-size 8192 --kv-commit 8192 --placement-strategy static)
run -v $M:/m -v /path/to/hf/DeepSeek-V4.1-Flash:/ck:ro "$IMG" "${common[@]}" \
    -p "$TEXT" --dump-tokens > "$OUT.tokens" 2>&1
IDS=$(grep -oE "[0-9]+(,[0-9]+)+" "$OUT.tokens" | tail -1)
echo "ids: $IDS"
[ -n "$IDS" ] || { echo "no ids"; tail -5 "$OUT.tokens"; exit 1; }
run -e AFF_NORM_TRACE=1 -v $M:/m -v /path/to/hf/DeepSeek-V4.1-Flash:/ck:ro "$IMG" \
    "${common[@]}" --dump-logits 10 --tokens "$IDS" > "$OUT.logits" 2>&1
echo "exit $?  trace lines: $(grep -c norm-trace "$OUT.logits")  -> $OUT.{tokens,logits}"
grep -v norm-trace "$OUT.logits" | tail -14
