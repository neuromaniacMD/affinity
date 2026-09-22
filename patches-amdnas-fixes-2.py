#!/usr/bin/env python3
"""aff-fork-patch2.py — affinity fork patch 2 (run inside ~/src/affinity-fork on amdnas): check EVERY kernel launch.
Why (2026-09-22 00:10): the process-poison on amdnas needs the placement mover AND a card packed to ~122 MiB of full
(probe 7: heat poisons 3/3, --placement-strategy static clean, --gpu-headroom-mib 2048 clean). Nothing in the logs — the
checked HIP calls abort loudly, so the failing operation is one that is never checked. 24 of the engine's 83 kernel
launches are never followed by hipGetLastError(); two of them are the plan builder and the plan-table PUBLISH
(moe_plan.hip). A publish launch dropped under memory pressure leaves the device table pointing an expert at a slab slot
the host bookkeeping then frees and hands to another expert: permanent wrong mapping = fluent garbage, 0 % acceptance.
This patch makes every previously unchecked launch fatal (AFF_HIP_CHECK) or a false return (moe_plan, whose callers abort).
Idempotent: skips a file whose marker is already present."""
import sys
MARK = "amdnas-fixes launch-check"
def edit(path, pairs):
    s = open(path).read()
    if MARK in s: print(f"{path}: already patched"); return
    for old, new, cnt in pairs:
        n = s.count(old)
        if n != cnt: sys.exit(f"ABORT {path}: anchor occurs {n}x (expected {cnt}):\n{old[:110]}")
        s = s.replace(old, new)
    s = s.replace("\n", "\n", 1)  # no-op; keep file otherwise identical
    s += f"\n// {MARK}: every hipLaunchKernelGGL in this file is followed by a launch-error check (2026-09-22).\n"
    open(path, "w").write(s); print(f"{path}: patched")

edit("src/gpu/moe_plan.hip", [
 ("  hipLaunchKernelGGL(plc_publish_kernel, dim3(1), dim3(kPlcPatchMax), 0, (hipStream_t)stream, plc, p);\n  return true;",
  "  hipLaunchKernelGGL(plc_publish_kernel, dim3(1), dim3(kPlcPatchMax), 0, (hipStream_t)stream, plc, p);\n"
  "  // amdnas-fixes: a publish launch that never ran leaves the DEVICE table stale while the host bookkeeping\n"
  "  // (offset_of_, pool binding, free lists) advances — the freed slot is then handed to another expert and the table\n"
  "  // points two experts at one slot for the life of the process. The caller aborts on false; that is the right outcome.\n"
  "  return hipGetLastError() == hipSuccess;", 1),
 ("  hipLaunchKernelGGL(moe_plan_kernel, dim3(1), dim3(kPlanThreads), lds, (hipStream_t)stream, a);\n  return true;",
  "  hipLaunchKernelGGL(moe_plan_kernel, dim3(1), dim3(kPlanThreads), lds, (hipStream_t)stream, a);\n"
  "  return hipGetLastError() == hipSuccess;   // amdnas-fixes: an unbuilt plan must not be dispatched", 1),
])
edit("src/gpu/expert_kernel.hip", [
 ("0, (hipStream_t)stream, j0, j1, cb, x, wscale, m.rows, m.cols)\n",
  "0, (hipStream_t)stream, j0, j1, cb, x, wscale, m.rows, m.cols); AFF_HIP_CHECK(hipGetLastError())\n", 1),
 ("(hipStream_t)stream, a, x, x_stride);", "(hipStream_t)stream, a, x, x_stride); AFF_HIP_CHECK(hipGetLastError());", 1),
 ("(hipStream_t)stream, W, S, x, nw, out, xn, rows, cols, sc, eps)\n",
  "(hipStream_t)stream, W, S, x, nw, out, xn, rows, cols, sc, eps); AFF_HIP_CHECK(hipGetLastError())\n", 1),
 ("(hipStream_t)stream, p, weights + off, n, cb, y, rows, cols)\n",
  "(hipStream_t)stream, p, weights + off, n, cb, y, rows, cols); AFF_HIP_CHECK(hipGetLastError())\n", 2),
 ("(hipStream_t)stream, p, weights + off, n, cb, rows, cols)\n",
  "(hipStream_t)stream, p, weights + off, n, cb, rows, cols); AFF_HIP_CHECK(hipGetLastError())\n", 1),
 ("cols, y_bf16 ? 1 : 0)\n", "cols, y_bf16 ? 1 : 0); AFF_HIP_CHECK(hipGetLastError())\n", 1),
])
edit("src/gpu/indexer_gpu.hip", [
 ("                         sp->state);\n", "                         sp->state);\n      AFF_HIP_CHECK(hipGetLastError());\n", 1),
])
edit("src/gpu/tp_allreduce.hip", [
 ("                       ctx.timeout_dev, n4, slot4, peer4, (int)r);\n",
  "                       ctx.timeout_dev, n4, slot4, peer4, (int)r);\n    AFF_HIP_CHECK(hipGetLastError());\n", 1),
 ("                       ctx.timeout_dev, n, slot_bytes, peer_bytes, scale_off, (int)r);\n",
  "                       ctx.timeout_dev, n, slot_bytes, peer_bytes, scale_off, (int)r);\n    AFF_HIP_CHECK(hipGetLastError());\n", 1),
])
print("patch 2 done")
