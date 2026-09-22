#!/bin/bash
# worker.sh — distributed affinity expert encode for DeepSeek-V4.1-Flash (2026-09-22).
# Runs on amdnas (the hub: holds the checkpoint + collects blobs) and on any number of helpers.
# Each layer is claimed with an atomic `mkdir` on the hub, so faster machines simply take more.
#   usage: worker.sh THREADS
# Inputs every worker must share byte-for-byte: the binary, v41cb.raw, the imatrix.
set -u
HUB=amdnas
HUBDIR=/path/to/aff-v41
CK=/path/to/hf/DeepSeek-V4.1-Flash
IMNAME=imatrix-dsv41-262144.gguf
T=${1:-16}
ME=$(hostname -s)
B=$HOME/aff-v41/bin/aff-quantize
ts() { date +%T; }

if [ "$ME" = amdnas ]; then
  LOCAL=1; SRC=$CK; OUT=$HUBDIR/experts; CB=$HUBDIR/v41cb.raw; IM=/mnt/models2/affinity/v41/imatrix/$IMNAME
  hub() { bash -c "$1"; }
else
  LOCAL=0; SRC=$HOME/aff-v41/src; OUT=$HOME/aff-v41/out; CB=$HOME/aff-v41/v41cb.raw; IM=$HOME/aff-v41/$IMNAME
  hub() { ssh -o BatchMode=yes $HUB "$1"; }
  mkdir -p "$SRC" "$OUT"
  rsync -a $HUB:$CK/config.json $HUB:$CK/model.safetensors.index.json "$SRC/" || exit 1
  rsync -a $HUB:$HUBDIR/v41cb.raw "$CB" || exit 1
  rsync -a $HUB:/mnt/models2/affinity/v41/imatrix/$IMNAME "$IM" || exit 1
fi
mkdir -p "$OUT"
hub "mkdir -p $HUBDIR/claims $HUBDIR/experts"
echo "$(ts) $ME worker start: threads $T, local=$LOCAL, cb $(sha256sum "$CB" | cut -c1-16), im $(stat -c %s "$IM")"

for l in $(seq 0 39); do
  hub "mkdir $HUBDIR/claims/L$l 2>/dev/null" || continue
  shard=$(printf "model-%05d-of-00048.safetensors" $((l + 3)))
  echo "$(ts) $ME claimed L$l ($shard)"
  if [ $LOCAL = 0 ]; then
    if ! rsync -a $HUB:$CK/$shard "$SRC/"; then
      echo "$(ts) $ME fetch L$l FAILED — releasing"; hub "rm -rf $HUBDIR/claims/L$l"; continue
    fi
  fi
  nice -n 10 "$B" --src "$SRC" --codebook-raw "$CB" --imatrix "$IM" --experts-dir "$OUT" \
       --layer-range $l:$((l + 1)) --threads $T > "$OUT/L$l.log" 2>&1
  rc=$?
  blob=$(printf "%s/L%03d.affx" "$OUT" $l)
  if [ $rc != 0 ] || [ ! -f "$blob" ]; then
    echo "$(ts) $ME encode L$l FAILED rc=$rc — releasing (log $OUT/L$l.log)"; tail -3 "$OUT/L$l.log"
    hub "rm -rf $HUBDIR/claims/L$l"; continue
  fi
  if [ $LOCAL = 0 ]; then
    rsync -a "$blob" $HUB:$HUBDIR/experts/ || { echo "$(ts) $ME push L$l FAILED — releasing"; hub "rm -rf $HUBDIR/claims/L$l"; continue; }
    rm -f "$SRC/$shard" "$blob"
  fi
  hub "echo '$ME $(ts) $(grep -o 'layers in [0-9]* s' $OUT/L$l.log)' > $HUBDIR/claims/L$l/done"
  echo "$(ts) $ME done L$l ($(grep -o 'in [0-9]* s' "$OUT/L$l.log" | tail -1))"
done
echo "$(ts) $ME worker exit: no unclaimed layers left"
