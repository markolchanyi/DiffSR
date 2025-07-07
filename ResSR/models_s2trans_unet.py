# s2_unet_global.py — memory‑light version (no external deps)
"""
Per‑voxel S²‑Transformer + 3‑D UNet with global tokens
=====================================================
• Input / output :  (B, 29, D, H, W)  — MRtrix SH basis (b0 + l = 0,2,4,6)
• Equivariance   :  exact SO(3) (icosahedral grid + sign‑mask)
• Dependencies   :  PyTorch ≥1.13, e3nn ≥0.5 (works on 0.5–0.7)
"""
from __future__ import annotations
import torch, torch.nn as nn
from functools import lru_cache
from e3nn import o3

# ---------------------------------------------------------------------------
# 1. Icosahedral grid helper
# ---------------------------------------------------------------------------
try:
    from e3nn.util.grid import icosahedral_sphere  # e3nn ≥0.7
except ImportError:
    def icosahedral_sphere(level: int = 2):  # fallback for old e3nn
        import numpy as np
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

# ---------------------------------------------------------------------------
# 3. Projection matrix SH→amplitude  (l=2,4,6 → 27 coeffs)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 4. S² Transformer (per voxel)
# ---------------------------------------------------------------------------
class S2Trans(nn.Module):
    def __init__(self, hidden=64, heads=8, level=3):
        super().__init__(); self.level=level
        n_dir=_proj(level,"cpu").shape[0]
        self.in_fc=nn.Linear(1+n_dir,hidden)
        block=nn.TransformerEncoderLayer(hidden,heads,hidden*4,batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(block,2)
        self.out_fc=nn.Linear(hidden,1+n_dir)
    def forward(self, c):  # c (..., 29)
        A = _proj(self.level, c.device)  # (N_dir, 27)
        b0 = c[..., :1]
        sh27 = torch.cat([c[..., 2:7], c[..., 7:16], c[..., 16:]], dim=-1)
        amp = sh27 @ A.T  # (..., N_dir)

        token = torch.cat([b0, amp], dim=-1)  # (..., 1 + N_dir)
        h = self.in_fc(token).unsqueeze(1)    # (..., 1, hidden)
        h = self.enc(h)                       # still (..., 1, hidden) but with MHA
        out_tok = self.out_fc(h.squeeze(1))   # (..., 1 + N_dir)

        b0o, amp_o = out_tok[..., :1], out_tok[..., 1:]
        sh_out = amp_o @ A                    # (..., 27)
        return torch.cat([b0o, c[..., 1:2], sh_out], dim=-1)

# ---------------------------------------------------------------------------
# 4.5 Voxel wrapper (apply S² block at each voxel)
# ---------------------------------------------------------------------------
class VoxelS2(nn.Module):
    def __init__(self, hidden=64):
        super().__init__(); self.core=S2Trans(hidden=hidden)
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
    def __init__(self,cin,cout): super().__init__(); self.net=nn.Sequential(nn.Conv3d(cin,cout,3,padding=1),nn.GELU(),nn.Conv3d(cout,cout,3,padding=1),nn.GELU())
    def forward(self,x): return self.net(x)

class UNet3d(nn.Module):
    def __init__(self,in_ch=29,base=192,depth=4):
        super().__init__(); self.enc=nn.ModuleList(); self.pool=nn.ModuleList(); self.dec=nn.ModuleList(); ch=in_ch
        for d in range(depth):
            out=base*2**d; self.enc.append(CBlock(ch,out)); ch=out; self.pool.append(nn.AvgPool3d(2)) if d<depth-1 else None
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
class S2UNetGlobal_MRtrix(nn.Module):
    def __init__(self, base=256):
        super().__init__(); self.register_buffer("sign",_sign.view(1,29,1,1,1))
        self.enc=VoxelS2(64); self.unet=UNet3d(29,base); self.dec=VoxelS2(64)
    def forward(self,x): x=x*self.sign; x=self.dec(self.unet(self.enc(x))); return x*self.sign

# ---------------------------------------------------------------------------
# 7. Smoke‑test
# ---------------------------------------------------------------------------
def _smoke():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    x=torch.randn(1,29,32,32,32,device=dev)
    y=S2UNetGlobal_MRtrix().to(dev)(x)
    print("🟢",x.shape,"→",y.shape)
if __name__=="__main__": _smoke()

