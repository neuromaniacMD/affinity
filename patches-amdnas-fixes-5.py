#!/usr/bin/env python3
"""aff-fork-patch5.py — affinity fork patch 5 (run inside ~/src/affinity-fork on amdnas): reset device history per request.
Probe 12 (2026-09-22 00:49): with every promotion AND demotion read back byte-exact, no slab/pool double-map, and GTT flat
(no driver relocation), the process STILL poisons 8/8 at default headroom. So the expert bytes and their mapping are sound;
what persists across requests is a corrupted ACTIVATION state. model.h:459 documents `BatchOps::compress_reset` as
"Discards the cross-chunk window for every layer. Called when a sequence is reset" — but nothing calls it. The compressor /
indexer history rings are position-indexed and the author argues a prefill from 0 overwrites every slot it later reads, yet
the first `hist` positions of a request pool over slots the PREVIOUS request's tail wrote; a NaN/garbage row there (from the
mover race) would poison every later request from its first token. This patch calls compress_reset from Model::init_state
(the per-request reset) when the op is bound. Correctness-neutral for healthy state; expected to turn the poison from
process-permanent into a single-request glitch if the persistence lives in those rings."""
import sys
MARK = "amdnas-fixes init-state compress_reset"
p = "src/engine/model.cpp"; s = open(p).read()
if MARK in s: print("already patched"); sys.exit(0)
old = '''void Model::init_state(SeqState* s, uint64_t max_pos) const {
  s->layer.assign(cfg_.n_layer, LayerState{});
  s->hc.assign((size_t)cfg_.hc_mult * cfg_.n_embd, 0.0f);
  s->pos = 0;'''
new = '''void Model::init_state(SeqState* s, uint64_t max_pos) const {
  s->layer.assign(cfg_.n_layer, LayerState{});
  s->hc.assign((size_t)cfg_.hc_mult * cfg_.n_embd, 0.0f);
  s->pos = 0;
  // amdnas-fixes init-state compress_reset: the device-side compressor/indexer history rings were never cleared between
  // requests (the op existed, nothing called it). A request's first `hist` positions pool over slots the previous request's
  // tail wrote; if one of those rows went bad, every later request inherits it. Clear them with the host state.
  if (bops_.compress_reset && bops_.ctx) (void)bops_.compress_reset(bops_.ctx);'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT init_state anchor {n}x")
s = s.replace(old, new)
open(p, "w").write(s); print("patch 5 applied")
