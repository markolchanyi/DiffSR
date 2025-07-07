"""
SO(3)-equivariant S²-Transformer (e3nn-0.5.6)
Token = amplitudes only (N_dir scalars)
Scalars b0 & l0 skip the block unchanged
"""

from __future__ import annotations
import torch
import torch.nn as nn
from functools import lru_cache
from e3nn import o3
from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from e3nn.nn import Gate


# ─────────  Icosahedral grid (fallback for e3nn-0.5)  ──────────
try:
    from e3nn.util.grid import icosahedral_sphere
except ImportError:
    def icosahedral_sphere(level=2):
        import numpy as np
        φ  = (1 + 5 ** 0.5) / 2
        v0 = np.array([
            [-1, φ, 0],[ 1, φ, 0],[-1,-φ, 0],[ 1,-φ, 0],
            [0,-1, φ],[ 0, 1, φ],[0,-1,-φ],[ 0, 1,-φ],
            [ φ, 0,-1],[ φ, 0, 1],[-φ, 0,-1],[-φ, 0, 1]])
        v0 /= np.linalg.norm(v0, 1, keepdims=True)
        f = np.array([
            [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
            [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
            [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
            [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]])
        v = v0.copy()
        for _ in range(level):
            vlist, flist, mid = v.tolist(), [], {}
            def m(a,b):
                k = tuple(sorted((a,b)))
                if k in mid: return mid[k]
                p = (v[a]+v[b])*.5; p/=np.linalg.norm(p)
                mid[k] = len(vlist); vlist.append(p); return mid[k]
            for a,b,c in f:
                ab,bc,ca = m(a,b), m(b,c), m(c,a)
                flist += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
            v,f = np.asarray(vlist), np.asarray(flist)
        return v


# ─────────  SH → amplitude projection (cached per device)  ─────
@lru_cache(maxsize=None)
def proj(level: int, device):
    dirs = torch.tensor(icosahedral_sphere(level),
                        dtype=torch.float32, device=device)
    Ys = [o3.spherical_harmonics(l, dirs, True) for l in (2,4,6)]
    return torch.cat(Ys, 1)            # (N_dir, 27)


# ─────────  Self tensor-product (x, x) wrapper  ─────────────────
class SelfTP(nn.Module):
    def __init__(self, in_irreps, out_irreps, *, dev):
        super().__init__()
        self.tp = FullyConnectedTensorProduct(
            in_irreps, in_irreps, out_irreps,
            internal_weights=True, shared_weights=True).to(dev)
    def forward(self, x): return self.tp(x, x)


# ─────────  Equivariant MLP core  ───────────────────────────────
def equiv_core(n_dir: int, h_s=32, h_v=8, *, dev):
    in_ir   = Irreps(f"{n_dir}x0e")        # amplitudes only
    s_mid   = Irreps(f"{h_s}x0e")
    g_mid   = Irreps(f"{h_v}x0e")
    v_mid   = Irreps(f"{h_v}x1e")

    pre_gate_ir  = s_mid + g_mid + v_mid   # output of tp1 / input of Gate
    post_gate_ir = s_mid + v_mid           # Gate output (gates removed)

    tp1  = SelfTP(in_ir,       pre_gate_ir,  dev=dev)
    gate = Gate(s_mid, [nn.GELU()],
                g_mid,
                [nn.Tanh()],
                v_mid).to(dev)
    tp2  = SelfTP(post_gate_ir, in_ir,       dev=dev)
    return nn.Sequential(tp1, gate, tp2)


# ─────────  S²-Transformer block  ───────────────────────────────
class S2Transformer(nn.Module):
    def __init__(self, level=3, h_s=32, h_v=8):
        super().__init__()
        self.level, self.h_s, self.h_v = level, h_s, h_v
        self.register_buffer("A", torch.empty(0))
        self.core: nn.Module | None = None

    def forward(self, sh29):                 # (..., 29)
        dev = sh29.device
        if self.A.numel() == 0 or self.A.device != dev:
            self.A.data = proj(self.level, dev)

        b0   = sh29[..., :1]
        l0   = sh29[..., 1:2]
        sh27 = torch.cat([sh29[..., 2:7],
                          sh29[..., 7:16],
                          sh29[..., 16:]], -1)

        amp = sh27 @ self.A.t()              # (..., N_dir)

        # (re)build core if N_dir changed
        if (self.core is None
            or amp.shape[-1] != self.core[0].tp.irreps_in1.dim):
            self.core = equiv_core(amp.shape[-1],
                                    self.h_s, self.h_v, dev=dev)

        amp = self.core(amp)                 # mix directions
        sh27o = amp @ self.A                 # back to SH
        return torch.cat([b0, l0, sh27o], -1)


# ─────────  Smoke-test  ─────────────────────────────────────────
if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x   = torch.randn(4, 29, device=dev, requires_grad=True)
    blk = S2Transformer(level=3).to(dev)
    y   = blk(x)
    print("token in :", x.shape[-1])
    print("token out:", y.shape[-1])
    print("CUDA?    :", torch.cuda.is_available())
    y.sum().backward()
    print("grad ‖x‖ :", x.grad.norm().item())

