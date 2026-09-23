#!/bin/bash
# The V4.1 performance squeeze, one server load an arm. Run ON amdnas with llama-swap STOPPED:
#
#   sudo systemctl stop llama-swap
#   bash ~/src/affinity-fork/tools/v41/squeeze.sh [image] [arms...]    # ~35 min for all four
#   sudo systemctl start llama-swap
#
# Arms (default: all, in this order):
#   backed  --kv-commit = --kv-size, so the KV grower never runs. Checks the 64K decode cliff is
#           gone (it was a grower retrying a growth the card could not hold; fixed in 0cdcf97, and
#           this flag avoids the grower altogether). Bench 16/48/64K + two-turn reference texts.
#   prefix  backed + --prefix-cache-dir. Two-turn at 16K and 64K: turn 2's TTFT is the win, and
#           its text must equal the `backed` arm's (greedy) or the restore changed the state.
#   demand  backed + --placement-strategy demand. Bench 16/48K against backed's rows.
#   chunk   backed + --prefill-chunk-size 2816. Bench 16/48K: prefill up, residency down.
#
# Every arm's server log is followed to the end, so the exit report (hit rate, prefix-cache
# counters) is kept — `run.sh serve` uses --rm, and the 09-22 bench lost it that way.
# ⚠️ Run as `bash squeeze.sh`, not ./squeeze.sh, if it ever lives on /mnt/models2 (SELinux).
set -uo pipefail
cd "$(dirname "$0")"
IMG=${1:-local/affinity:v41-seed}; shift || true
ARMS=${*:-backed prefix demand chunk}
OUT=/mnt/models2/affinity/v41/squeeze-$(date +%Y%m%d-%H%M)
mkdir -p "$OUT"
BASE=(--kv-size 69632 --kv-commit 69632 --engram /m/model.engram --engram-tables /ck
      --max-tokens-floor 0)
URL=http://127.0.0.1:8099/v1/chat/completions
echo "image $IMG ($(docker image inspect "$IMG" --format '{{.Id}}' | cut -c8-19)), arms: $ARMS, out $OUT"

arm() {   # arm <name> <extra flags...> -- <workload function>
  local name=$1; shift
  local flags=(); while [ "$1" != "--" ]; do flags+=("$1"); shift; done; shift
  local work=$1
  echo "=== $name: ${flags[*]}"
  sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  ./run.sh serve "$IMG" sq-$name "${BASE[@]}" "${flags[@]}" >/dev/null || { echo "$name: serve failed"; return; }
  docker logs -f sq-$name > "$OUT/$name.server.log" 2>&1 &
  local follower=$!; sleep 2
  grep -E "placement sizing|static placement|kv cache:" "$OUT/$name.server.log" | sed "s/^/  /"
  $work 2>&1 | tee "$OUT/$name.txt"
  docker stop -t 60 sq-$name >/dev/null 2>&1
  wait $follower 2>/dev/null
  grep -iE "hit rate|hit|resumed|cannot grow|restored" "$OUT/$name.server.log" | tail -8 | sed "s/^/  /"
}

w_backed() { ./run.sh bench n16k.txt n48k.txt n64k.txt
             URL=$URL python3 -u turn2.py /mnt/models2/affinity/v41/n16k.txt n16k
             URL=$URL python3 -u turn2.py /mnt/models2/affinity/v41/n64k.txt n64k; }
w_prefix() { URL=$URL python3 -u turn2.py /mnt/models2/affinity/v41/n16k.txt n16k
             URL=$URL python3 -u turn2.py /mnt/models2/affinity/v41/n64k.txt n64k; }
w_short()  { ./run.sh bench n16k.txt n48k.txt; }

for a in $ARMS; do
  case $a in
    backed) arm backed -- w_backed ;;
    prefix) mkdir -p /mnt/models2/affinity/v41/pcache
            arm prefix --prefix-cache-dir /m/pcache --prefix-cache-gib 64 -- w_prefix ;;
    demand) arm demand --placement-strategy demand -- w_short ;;
    chunk)  arm chunk --prefill-chunk-size 2816 -- w_short ;;
    *) echo "unknown arm $a" ;;
  esac
done
echo "SQUEEZE-DONE $OUT"
