#!/bin/bash
# Rate-capped resume of DeepSeek-V4.1-Flash (user 2026-09-22: "limit the download speed a little").
# PER_STREAM x STREAMS = total cap. Verifies every shard against the HF LFS sha256 at the end.
set -u
REPO=deepseek-ai/DeepSeek-V4.1-Flash; REV=dba1be0a40aa45a94ad051997016db3960a90277
DIR=$HOME/hf/DeepSeek-V4.1-Flash; PER_STREAM=${PER_STREAM:-75M}; STREAMS=${STREAMS:-4}
cd "$DIR" || exit 1
curl -s "https://huggingface.co/api/models/$REPO/tree/$REV?recursive=true" -o .tree.json
python3 - > .want.tsv <<PY
import json
for x in json.load(open(".tree.json")):
    if x["type"]=="file" and x["path"].endswith(".safetensors"):
        print(x["path"], x["size"], x.get("lfs",{}).get("oid",""))
PY
fetch() { f=$1; sz=$2
  [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$sz" ] && return 0
  curl -sSfL --retry 5 -C - --limit-rate "$PER_STREAM" -o "$f.part" "https://huggingface.co/$REPO/resolve/$REV/$f" && mv "$f.part" "$f" && echo "$(date +%T) got $f"; }
export -f fetch; export REPO REV PER_STREAM
awk "{print \$1, \$2}" .want.tsv | xargs -P "$STREAMS" -n 2 bash -c "fetch \"\$0\" \"\$1\""
echo "$(date +%T) download pass done; verifying sha256"
bad=0; while read f sz oid; do h=$(sha256sum "$f" | cut -d" " -f1); [ "$h" = "$oid" ] || { echo "BAD $f"; bad=$((bad+1)); }; done < .want.tsv
echo "$(date +%T) verify done: $bad bad of $(wc -l < .want.tsv)"
