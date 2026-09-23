#!/usr/bin/env python3
"""Engram's gate, checked against the reference module itself.

The hash and the table read are checked by aff-engramcheck; this is the third link -- wkv, the
key/value split, the gate and the residual write. It builds the reference `Engram` with `embed`
patched to hand back fixed rows (the real one wants a 91.6 GiB table) and the checkpoint's REAL
q_weight/k_weight/wkv, then compares it against the same formula written out in numpy, which is
what the HIP kernel transcribes.

Expected: max |ref - mine| ~1.6e-07, with gates spread across the four hc copies rather than
sitting at a fixed point -- a formula error that collapsed the gate would still "agree" at 0.5.

    ~/aff-v41/venv/bin/python engram_gatecheck.py
"""
import sys, numpy as np, torch
sys.path.insert(0,'/path/to/hf/DeepSeek-V4.1-Flash/inference')
from safetensors import safe_open
ck='/path/to/hf/DeepSeek-V4.1-Flash/'
L, SH = 1, 'model-00047-of-00048.safetensors'
DIM, HC, NCOL, HD = 5120, 4, 24, 256

with safe_open(ck+SH,'pt') as f:
    q = f.get_tensor(f'layers.{L}.engram.q_weight').float()      # [4,5120]
    k = f.get_tensor(f'layers.{L}.engram.k_weight').float()
    W = f.get_tensor(f'layers.{L}.engram.wkv.weight')            # fp8 [25600,6144]
    S = f.get_tensor(f'layers.{L}.engram.wkv.scale')             # [800,192]
# dequantise wkv: 32x32 blocks, E8M0 scales
Wf = W.float().numpy()
Ss = S.view(torch.uint8).numpy().astype(np.int32)
sc = np.ldexp(1.0, Ss-127)                                        # [800,192]
Wf = Wf.reshape(800,32,192,32) * sc[:,None,:,None]
Wf = Wf.reshape(25600,6144)

rng = np.random.default_rng(3)
rows = rng.standard_normal(NCOL*HD).astype(np.float32)*0.3        # the 'looked-up' table rows
x    = rng.standard_normal((HC,DIM)).astype(np.float32)*0.8       # the hidden state
eps  = 1e-6

# ---- the reference's own module, with embed patched to hand back our rows ----
sys.path.insert(0,'/path/to/aff-v41')
import kernel_shim
sys.modules['kernel']=kernel_shim
import model as M
M.world_size, M.rank = 1, 0
from engram import EngramLayout
class A: pass
a=A(); a.dim=DIM; a.hc_mult=HC; a.norm_eps=eps
class Lay: num_embeddings=[0,0]; head_dim=HD; max_ngram_size=4; n_heads=8; layer_ids=(1,14)
eng = M.Engram.__new__(M.Engram)
torch.nn.Module.__init__(eng)
eng.layer_id=L; eng.layer_hash_index=0; eng.dim=DIM; eng.hc_mult=HC; eng.clamp_value=1e-6
eng.eps=eps
eng.q_weight=torch.nn.Parameter(q); eng.k_weight=torch.nn.Parameter(k)
eng.embed = lambda ids: torch.tensor(rows).reshape(1,1,NCOL,HD)
eng.wkv   = lambda v: torch.tensor(v.numpy() @ Wf.T.astype(np.float32))
with torch.no_grad():
    ref = eng.forward(torch.tensor(x).reshape(1,1,HC,DIM), None).numpy()[0,0]

# ---- my reading of it, in numpy ----
kv = rows @ Wf.T.astype(np.float32)
key, value = kv[:HC*DIM].reshape(HC,DIM), kv[HC*DIM:]
w = (q*k).numpy()
rstd = (1/np.sqrt((x*x).mean(-1)+eps)) * (1/np.sqrt((key*key).mean(-1)+eps))
dot  = (x*w*key).sum(-1) * rstd / np.sqrt(DIM)
gate = 1/(1+np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(dot),1e-6)), dot)))
mine = x + gate[:,None]*value[None,:]

print('gate per hc copy:', gate)
print('max |ref - mine| =', np.abs(ref-mine).max())
print('ref[0,:4]  =', ref[0,:4])
print('mine[0,:4] =', mine[0,:4])
