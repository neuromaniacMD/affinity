#!/bin/bash
# Needle with engram OFF vs ON, same image, same server flags otherwise. Run with llama-swap STOPPED.
#   bash needle_ab.sh <image> [off|on ...]
set -uo pipefail
cd "$(dirname "$0")"
IMG=${1:?image}; shift; ARMS=${*:-off on}
M=/mnt/models2/affinity/v41; OUT=$M/needle-ab-$(date +%H%M); mkdir -p $OUT
for a in $ARMS; do
  E=(); [ "$a" = on ] && E=(--engram /m/model.engram --engram-tables /ck)
  ./run.sh serve "$IMG" nab-$a --kv-size 69632 --kv-commit 69632 --max-tokens-floor 0 "${E[@]}" >/dev/null
  echo "=== engram $a"
  URL=http://127.0.0.1:8099/v1/chat/completions python3 -u needle.py $M/n4k.txt $M/n16k.txt | tee $OUT/$a.txt
  docker logs nab-$a > $OUT/$a.server.log 2>&1; docker stop -t 30 nab-$a >/dev/null 2>&1
done
echo "NEEDLE-AB-DONE $OUT"
