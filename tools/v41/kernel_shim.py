"""CPU stand-ins for DeepSeek-V4.1's TileLang kernels, so the checkpoint's own inference/model.py
runs on a CPU box and can produce golden vectors for the affinity port.

Injected as the module named `kernel` before importing model.py. Arithmetic-only kernels
(act_quant / fp8_gemm / fp4_gemm) are replaced by a dequantise-then-matmul path in float32, which is
MORE precise than the GPU kernels — golden vectors are compared with a tolerance, not bit-exactly.
The two kernels that are model math, `hc_split_sinkhorn` and `sparse_attn`, are reimplemented
statement-for-statement from kernel.py (see the comments) because a wrong one would silently produce
a plausible but wrong reference.
"""
import torch
import torch.nn.functional as F

FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def _e8m0(scale: torch.Tensor) -> torch.Tensor:
    """UE8M0 byte -> 2^(e-127)."""
    e = scale.view(torch.uint8).to(torch.int32)
    return torch.ldexp(torch.ones_like(e, dtype=torch.float32), e - 127)


def dequant_fp8(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """[N,K] fp8_e4m3 with [ceil(N/b), ceil(K/b)] UE8M0 block scales -> float32."""
    x = w.to(torch.float32)
    sc = _e8m0(s) if s.dtype == torch.float8_e8m0fnu else s.to(torch.float32)
    sc = sc.repeat_interleave(block, 0).repeat_interleave(block, 1)[: x.size(0), : x.size(1)]
    return x * sc


def dequant_fp4(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """[N,K//2] packed fp4_e2m1 (low nibble first) with [N, K/32] UE8M0 scales -> float32 [N,K]."""
    b = w.view(torch.uint8)
    lo, hi = b & 0xF, (b >> 4) & 0xF
    x = torch.stack([FP4_LUT[lo.long()], FP4_LUT[hi.long()]], dim=-1).flatten(-2)
    sc = _e8m0(s).repeat_interleave(block, -1)[..., : x.size(-1)]
    return x * sc


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Pass the activation through in float; the shimmed GEMMs ignore the scale."""
    if inplace:
        return x
    return x.float(), None


def fp4_act_quant(x, block_size=32, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    if inplace:
        return x
    return x.float(), None


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """C[M,N] = A[M,K] @ B[N,K]^T."""
    return F.linear(a.float(), dequant_fp8(b, b_s, block_size)).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    return F.linear(a.float(), dequant_fp4(b, b_s)).to(torch.get_default_dtype())


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Statement-for-statement from hc_split_sinkhorn_kernel: pre = sigmoid(mix*s0 + base) + eps,
    post = 2*sigmoid(mix*s1 + base), comb = row-softmax(mix*s2 + base) + eps then alternating
    column/row normalisation (the first column pass, then iters-1 row+column passes)."""
    *lead, _ = mixes.shape
    m = mixes.float().view(-1, (2 + hc_mult) * hc_mult)
    base, s = hc_base.float(), hc_scale.float()
    pre = torch.sigmoid(m[:, :hc_mult] * s[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(m[:, hc_mult:2 * hc_mult] * s[1] + base[hc_mult:2 * hc_mult])
    comb = (m[:, 2 * hc_mult:] * s[2] + base[2 * hc_mult:]).view(-1, hc_mult, hc_mult)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return (pre.view(*lead, hc_mult), post.view(*lead, hc_mult), comb.view(*lead, hc_mult, hc_mult))


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Multi-query attention over gathered indices, with the sink added to the denominator only.
    From sparse_attn_kernel: index -1 = masked (score -inf); a row with no valid index must give an
    all-zero output rather than NaN, which the kernel gets from its finite -1e30 running max."""
    b, s, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    gathered = kv.gather(1, idx.clamp_min(0).view(b, -1, 1).expand(-1, -1, d)).view(b, s, -1, d)
    scores = torch.einsum("bshd,bskd->bshk", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    e = torch.exp(scores - mx)
    e = torch.where(valid.unsqueeze(2), e, torch.zeros_like(e))
    denom = e.sum(-1, keepdim=True) + torch.exp(attn_sink.float().view(1, 1, h, 1) - mx)
    o = torch.einsum("bshk,bskd->bshd", e / denom, gathered.float())
    return o.to(q.dtype)
