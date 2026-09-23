#!/bin/bash
# The V4.1 lane on amdnas: one shot, or a server to benchmark against.
#
# Replaces the half-dozen /tmp wrappers this port was driven by, which were one reboot from gone.
# Everything it hard-codes is amdnas-specific and is listed at the top so it can be changed in one
# place.
#
#   run.sh once   <image> <prompt-file> <n> [extra flags...]
#   run.sh serve  <image> <name> [extra flags...]        # background, port 8099
#   run.sh bench  <prompt-file...>                       # against a running serve
#
# ⚠️ --security-opt label=disable is NOT optional on this SELinux host: without it the container's
#    read of the model is Permission denied.
# ⚠️ Engram is OFF unless asked for. Add:  --engram /m/model.engram --engram-tables /ck
#    Without it layers 1 and 14 contribute nothing, which is not the released model.
# ⚠️ --kv-size sizes the indexer key cache, the admission mask and the candidate mask from cache
#    CAPACITY, not from the context in use. The 1M default costs ~2 GiB a card against a 64K-sized
#    69632, which is 302 more resident experts a card. Size it to the context you actually want.
set -euo pipefail

MODELS=/mnt/models2/affinity/v41
CKPT=/path/to/hf/DeepSeek-V4.1-Flash
PORT=8099
POOL="--host-pool-mib 138000 --host-pool-reserve-mib 512"

docker_common=(--device=/dev/kfd --device=/dev/dri --security-opt label=disable
               --network host --ipc host --ulimit memlock=-1:-1 --shm-size 16g
               -e HIP_VISIBLE_DEVICES=0,1 -v "$MODELS":/m -v "$CKPT":/ck:ro)
engine_common=(-m /m/model.aff --tokenizer /ck/tokenizer.json --dspark off $POOL --ui plain)

mode=${1:?usage: run.sh once|serve|bench ...}; shift

case "$mode" in
  once)
    img=$1; file=$2; n=$3; shift 3
    exec docker run --rm "${docker_common[@]}" "$img" "${engine_common[@]}" \
      --prompt-file "/m/$file" -n "$n" "$@"
    ;;
  serve)
    img=$1; name=$2; shift 2
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run --rm -d --name "$name" "${docker_common[@]}" "$img" "${engine_common[@]}" \
      --port "$PORT" "$@"
    echo "waiting for $name on :$PORT (the pool populate alone is ~90 s)"
    until curl -s --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do sleep 10; done
    echo "ready"
    ;;
  bench)
    # Prompt names are resolved against $MODELS when they are not paths that exist here. The
    # fixtures live beside the model, not beside this script, and `./run.sh bench n4k.txt` from the
    # tools directory is the obvious thing to type — it should not be the thing that fails.
    args=()
    for f in "$@"; do
      if [ -e "$f" ]; then args+=("$f")
      elif [ -e "$MODELS/$f" ]; then args+=("$MODELS/$f")
      else echo "run.sh: no such prompt: $f (looked here and in $MODELS)" >&2; exit 2; fi
    done
    exec env URL="http://127.0.0.1:$PORT/v1/chat/completions" NGEN="${NGEN:-64}" \
      python3 "$(dirname "$0")/bench_pp_tg.py" "${args[@]}"
    ;;
  *) echo "usage: run.sh once|serve|bench ..." >&2; exit 2 ;;
esac
