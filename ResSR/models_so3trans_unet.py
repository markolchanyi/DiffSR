"""
Per‑voxel SO3 Transformer + 3‑D UNet with global tokens
"""
from __future__ import annotations
import torch, torch.nn as nn
from functools import lru_cache
from e3nn import o3
from e3nn.o3 import Irreps
from e3nn.nn import Gate
from e3nn.o3 import Linear
import numpy as np

# ---------------------------------------------------------------------------
# 1. Icosahedral grid helper
# ---------------------------------------------------------------------------
try:
    from e3nn.util.grid import icosahedral_sphere  # e3nn ≥0.7
except ImportError:
    def icosahedral_sphere(level: int = 2):  # fallback for old e3nn
        phi = (1 + 5 ** 0.5) / 2
        verts = np.array([
            [-1,  phi, 0],[ 1,  phi, 0],[-1, -phi, 0],[ 1, -phi, 0],
            [0, -1,  phi],[ 0,  1,  phi],[0, -1, -phi],[ 0,  1, -phi],
            [ phi, 0, -1],[ phi, 0,  1],[-phi,0, -1],[-phi,0,  1]])
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        faces = np.array([
            [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
            [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
            [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
            [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]])
        for _ in range(level):
            vlist, flist, mid = verts.tolist(), [], {}
            def m(a,b):
                key = tuple(sorted((a,b)))
                if key in mid: return mid[key]
                v = (verts[a]+verts[b])*.5; v/=np.linalg.norm(v)
                mid[key] = len(vlist); vlist.append(v); return mid[key]
            for a,b,c in faces:
                ab,bc,ca = m(a,b), m(b,c), m(c,a)
                flist += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
            verts, faces = np.asarray(vlist), np.asarray(flist)
        return verts

# ---------------------------------------------------------------------------
# 2. MRtrix ↔ canonical sign mask  ( (-1)^m )
# ---------------------------------------------------------------------------
_sign = torch.tensor([1,1]+[(-1)**m for l in (2,4,6) for m in range(-l,l+1)],dtype=torch.float32)  # (29,)

### cached (since only need once) projection from SH to amplitudes
@lru_cache(maxsize=1)
def _proj(level:int,device):
    dirs = torch.tensor(icosahedral_sphere(level),dtype=torch.float32,device=device)
    Ys=[]
    for l in (2,4,6):
        try:
            Y=o3.spherical_harmonics(l,dirs,normalization="component")
        except TypeError:
            Y=o3.spherical_harmonics(l,dirs,normalize=True)
        Ys.append(Y)
    return torch.cat(Ys,1)   # (N,27)



# ── quadratic (x ⊗ x) wrapper ───────────────────────────────────
class SelfTP(nn.Module):
    def __init__(self, in_irreps, out_irreps, *, dev):
        super().__init__()
        self.tp = FullyConnectedTensorProduct(
            in_irreps, in_irreps, out_irreps,
            internal_weights=True, shared_weights=True
        ).to(dev)

    def forward(self, x):          # quadratic map
        return self.tp(x, x)


# ── equivariant core with built-in residual ─────────────────────
def EquivCore(n_dir, h_s=32, h_v=0, *, dev):
    in_ir = Irreps(f"{n_dir}x0e")      # input amplitudes (scalars)
    s_mid = Irreps(f"{h_s}x0e")        # always keep the scalar trunk

    # ---------- scalar-only variant -------------------------------
    if h_v == 0:
        pre_ir  = s_mid                # tp1 output
        post_ir = s_mid                # gate output

        tp1  = SelfTP(in_ir, pre_ir, dev=dev)
        ln   = nn.LayerNorm(pre_ir.dim, elementwise_affine=False).to(dev)

        gate = Gate(                   # Gate needs 5 args even if some empty
            s_mid, [nn.GELU()],        # scalars + activation
            Irreps(""), [],            # no gate-scalars
            Irreps("")                 # no gated vectors
        ).to(dev)

        tp2  = SelfTP(post_ir, in_ir, dev=dev)

    # ---------- full scalar+vector variant ------------------------
    else:
        g_mid = Irreps(f"{h_v}x0e")
        v_mid = Irreps(f"{h_v}x1e")

        pre_ir  = s_mid + g_mid + v_mid
        post_ir = s_mid + v_mid

        tp1  = SelfTP(in_ir, pre_ir, dev=dev)
        ln   = nn.LayerNorm(pre_ir.dim, elementwise_affine=False).to(dev)

        gate = Gate(
            s_mid, [nn.GELU()],
            g_mid, [nn.Tanh()],
            v_mid
        ).to(dev)

        tp2  = SelfTP(post_ir, in_ir, dev=dev)

    # ------------- residual wrapper (with scaling) ----------------
    class Residual(nn.Module):
        def __init__(self, f, n_dir):
            super().__init__()
            self.f = f
            self.alpha = nn.Parameter(torch.tensor(0.1))
            self.n_dir = n_dir         # SO3Transformer needs this
        def forward(self, x):
            return x + self.alpha * self.f(x)

    core = nn.Sequential(tp1, ln, gate, tp2)
    return Residual(core, n_dir).to(dev)




# ─────────  SO3-Transformer block  ───────────────────────────────
class SO3Transformer(nn.Module):
    def __init__(self, level=3, h_s=32, h_v=0):
        super().__init__()
        self.level, self.h_s, self.h_v = level, h_s, h_v
        self.register_buffer("A", torch.empty(0))
        self.core: nn.Module | None = None

    def forward(self, sh29):                 # (..., 29)
        dev = sh29.device
        if self.A.numel() == 0 or self.A.device != dev:
            self.A.data = _proj(self.level, dev)

        b0   = sh29[..., :1]
        l0   = sh29[..., 1:2]
        sh27 = torch.cat([sh29[..., 2:7], sh29[..., 7:16], sh29[..., 16:]], -1)

        amp = sh27 @ self.A.T              # (..., N_dir)

        # rebuild core if N_dir changed
        if self.core is None or amp.shape[-1] != self.core.n_dir:
            self.core = EquivCore(amp.shape[-1], self.h_s, self.h_v, dev=dev)

        amp = self.core(amp)                 # mix directions
        sh27_out = amp @ self.A                 # back to SH

        return torch.cat([b0, l0, sh27_out], -1)



# old s2 wrapper
class VoxelSO3(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.core=SO3Transformer(level=2, h_s=64, h_v=0)

    def forward(self,x):  # x (B,29,D,H,W)
        B,C,D,H,W=x.shape
        x=x.permute(0,2,3,4,1).reshape(-1,C)
        x=self.core(x)
        return x.view(B,D,H,W,C).permute(0,4,1,2,3).contiguous()



# ---------------------------------------------------------------------------
# 5. Small 3‑D UNet with global tokens
# ---------------------------------------------------------------------------
class GTok(nn.Module):
    """Global token block: aggregates global context and broadcasts it back"""
    def __init__(self, ch: int, nt: int = 8):
        super().__init__()
        self.tok  = nn.Parameter(torch.randn(1, nt, ch))   # (1, T, C)
        self.attn = nn.MultiheadAttention(ch, 4, batch_first=True)
        self.ln   = nn.LayerNorm(ch)

    def forward(self, x):  # x: (B,C,D,H,W)
        B, C, D, H, W = x.shape
        seq = x.flatten(2).permute(0, 2, 1)   # (B, N, C)
        tok = self.tok.expand(B, -1, -1)      # (B, T, C)
        tok, _ = self.attn(tok, seq, seq)     # global tokens attend to sequence
        g = self.ln(tok.mean(dim=1))          # (B, C) global summary
        g = g.view(B, C, 1, 1, 1)             # broadcast
        return x + g


class CBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(cin,cout,3,padding=1),nn.GELU(),nn.Conv3d(cout,cout,3,padding=1),nn.GELU())
    def forward(self,x): return self.net(x)


class UNet3d(nn.Module):
    def __init__(self,in_ch=29,base=192,depth=4):
        super().__init__(); self.enc=nn.ModuleList(); self.pool=nn.ModuleList(); self.dec=nn.ModuleList(); ch=in_ch
        for d in range(depth):
            out=base*2**d
            self.enc.append(CBlock(ch,out))
            ch=out
            self.pool.append(nn.AvgPool3d(2)) if d<depth-1 else None
        self.gtok=GTok(ch)
        for d in reversed(range(depth-1)):
            out=base*2**d; self.dec.append(nn.Sequential(nn.ConvTranspose3d(ch,out,2,2),nn.GELU(),CBlock(out,out))); ch=out
        self.final=nn.Conv3d(ch,in_ch,1)
    def forward(self,x):
        skips=[]; h=x
        for enc,pool in zip(self.enc[:-1],self.pool): h=enc(h); skips.append(h); h=pool(h)
        h=self.enc[-1](h); h=self.gtok(h)
        for dec in self.dec: h=dec(h); h=h+skips.pop()
        return self.final(h)

# ---------------------------------------------------------------------------
# 6. Full model
# ---------------------------------------------------------------------------
class SO3UNetGlobal(nn.Module):
    def __init__(self, base=256):
        super().__init__()
        self.register_buffer("sign",_sign.view(1,29,1,1,1))
        self.enc=VoxelSO3()
        self.unet=UNet3d(29,base)
        self.dec=VoxelSO3()

    def forward(self,x):
        x=x*self.sign
        x=self.dec(self.unet(self.enc(x)))
        return x*self.sign

# ---------------------------------------------------------------------------
# 7. Smoke‑test
# ---------------------------------------------------------------------------
def _smoke():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    x=torch.randn(1,29,32,32,32,device=dev)
    y=SO3UNetGlobal().to(dev)(x)
    print("🟢",x.shape,"→",y.shape)
if __name__=="__main__": _smoke()

