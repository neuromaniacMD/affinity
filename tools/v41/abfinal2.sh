#!/bin/bash
# Fair A/B for the V4.1 lane flags: every arm gets the SAME request sequence on a server of its own,
# so page-cache warmth cannot favour whichever arm ran after a warm-up (the squeeze.sh arms could).
# Long-answer fixtures (l*.txt: same depths as n*.txt, a question that wants a detailed answer), so
# tg is measured over NGEN real tokens instead of an 8-token needle answer.
#   bash abfinal.sh <image> <arm> [<arm> ...]      arms: base demand chunkN kv262k kv262kcN
# Run with llama-swap STOPPED.
set -uo pipefail
cd "$(dirname "$0")"
IMG=${1:?image}; shift
M=/mnt/models2/affinity/v41; OUT=$M/abfinal-$(date +%H%M); mkdir -p $OUT
for a in "$@"; do
  KV=69632; F=()
  case $a in
    base) ;;
    demand) F=(--placement-strategy demand) ;;
    chunk[0-9]*) F=(--prefill-chunk-size ${a#chunk}) ;;
    kv262kc[0-9]*) KV=262144; F=(--prefill-chunk-size ${a#kv262kc}) ;;
    kv262k) KV=262144 ;;
    kv131kc[0-9]*) KV=131072; F=(--prefill-chunk-size ${a#kv131kc}) ;;
    *) echo "unknown arm $a"; continue ;;
  esac
  sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  ./run.sh serve "$IMG" ab-$a --kv-size $KV --kv-commit $KV --max-tokens-floor 0 \
      --engram /m/model.engram --engram-tables /ck "${F[@]}" >/dev/null || { echo "$a: serve failed"; continue; }
  docker logs ab-$a 2>&1 | grep -E "static placement|kv cache:" | sed "s/^/  /"
  echo "=== $a (kv $KV ${F[*]})"
  NGEN=16 ./run.sh bench l4k.txt > /dev/null 2>&1           # warm-up, discarded
  NGEN=192 ./run.sh bench l16k.txt l64k.txt | tee $OUT/$a.txt
  docker logs ab-$a > $OUT/$a.server.log 2>&1; docker stop -t 30 ab-$a >/dev/null 2>&1
done
echo "ABFINAL-DONE $OUT"
