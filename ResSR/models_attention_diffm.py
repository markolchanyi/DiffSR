# attn_unet_partial_diffusion.py
# ---------------------------------------------------------------
# 3-D Attention U-Net backbone (2-level) + 4-step DDPM refiner
# Input / Output tensor shape : (B, 29, D, H, W)
# ---------------------------------------------------------------

from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------
# sinusoidal timestep embedding (dim = 128)
# ----------------------------------------------------------------
def t_embed(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype)
        * (-math.log(10_000.0) / (half - 1))
    )
    emb = torch.cat([torch.sin(t[:, None] * freqs[None]),
                     torch.cos(t[:, None] * freqs[None])], 1)
    if dim % 2: emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)

# ----------------------------------------------------------------
# small Conv-GN-GELU
# ----------------------------------------------------------------
class ConvGNGLU(nn.Module):
    def __init__(self, c_in, c_out, ks=3, stride=1):
        super().__init__()
        self.c = nn.Conv3d(c_in, c_out, ks, stride, ks // 2)
        self.n = nn.GroupNorm(8, c_out, affine=True)
        self.a = nn.GELU()
    def forward(self, x):
        return self.a(self.n(self.c(x)))

#class ConvGNGLU(nn.Module):
#    def __init__(self, c_in, c_out, ks=3, stride=1):
#        super().__init__()
#        self.c = nn.Conv3d(c_in, c_out, ks, stride, ks // 2)
        #self.n = nn.GroupNorm(8, c_out, affine=True)
#        self.a = nn.GELU()
#    def forward(self, x): return self.a(self.c(x))

# ----------------------------------------------------------------
# shifted-window 3-D attention (single block, Swin-lite)
# ----------------------------------------------------------------
class WindowAttn3D(nn.Module):
    def __init__(self, dim, heads=4, ws=8):
        super().__init__()
        self.dim, self.h, self.ws = dim, heads, ws
        self.scale = (dim // heads) ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        size = 2 * ws - 1
        self.rel = nn.Parameter(torch.zeros(heads, size, size, size))
        nn.init.trunc_normal_(self.rel, .02)

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        ws, h, s = self.ws, self.h, self.ws // 2
        # cyclic shift
        x = torch.roll(x, (-s, -s, -s), (2, 3, 4))
        # partition windows (B nD nH nW N C)
        x = (x.reshape(B, C, D // ws, ws, H // ws, ws, W // ws, ws)
               .permute(0, 2, 4, 6, 3, 5, 7, 1)   # (B nD nH nW ws ws ws C)
               .reshape(-1, ws ** 3, C))          # (B*numW, N, C)
        qkv = self.qkv(x).reshape(-1, ws ** 3, 3, h, C // h)
        q, k, v = qkv.unbind(2)                   # (B*numW, N, h, d)
        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]  # (B*numW,h,N,d)
        attn = (q * self.scale) @ k.transpose(-2, -1)         # (B*numW,h,N,N)

        # relative positional bias
        coords = torch.stack(torch.meshgrid([torch.arange(ws)] * 3, indexing='ij'))
        coords = coords.flatten(1)
        rel = coords[:, :, None] - coords[:, None, :] + ws - 1  # (3,N,N)
        attn = attn + self.rel[:, rel[0], rel[1], rel[2]]       # (h,N,N)
        attn = attn.softmax(-1)

        out = (attn @ v).transpose(2, 3)   # (B*numW,h,d,N)
        out = out.transpose(1, 2).reshape(-1, ws ** 3, C)  # (B*numW,N,C)
        out = self.proj(out)
        # merge windows & reverse shift
        out = (out.reshape(B, D // ws, H // ws, W // ws, ws, ws, ws, C)
                    .permute(0, 7, 1, 4, 2, 5, 3, 6)
                    .contiguous()
                    .reshape(B, C, D, H, W))
        return torch.roll(out, (s, s, s), (2, 3, 4))

# ----------------------------------------------------------------
# encoder / decoder blocks
# ----------------------------------------------------------------
class Down(nn.Module):
    def __init__(self, ci, co): super().__init__(); self.m = nn.Sequential(
        ConvGNGLU(ci, co, 3, 2), ConvGNGLU(co, co))
    def forward(self, x): return self.m(x)

class Up(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.cv = ConvGNGLU(ci, co)
    def forward(self, x): return self.cv(self.up(x))

# ----------------------------------------------------------------
# backbone Attention U-Net (29->29)
# ----------------------------------------------------------------
class AttnUNet3D(nn.Module):
    """
    Stronger 3-D Attention U-Net
    • base = 96 channels
    • two down-sampling levels  (↓2, ↓4)  →  RF ≈ 80 mm at 1.25 mm vox
    • two shifted-window attention blocks (mid + deep)
    """
    def __init__(self, ch: int = 29, base: int = 96, ws: int = 4):
        super().__init__()

        # ── level-0 (full-res) ─────────────────────────────────────────────
        self.enc0 = ConvGNGLU(ch, base)                       # (B, 96)

        # ── level-1 (↓2) ───────────────────────────────────────────────────
        self.down1 = Down(base, base * 2)                     # (B,192)
        self.attn1 = WindowAttn3D(base * 2, heads=6, ws=ws)   # first attn

        # ── level-2 (↓4) ───────────────────────────────────────────────────
        self.down2 = Down(base * 2, base * 4)                 # (B,384)
        self.bott  = nn.Sequential(                           # deep attn
            WindowAttn3D(base * 4, heads=8, ws=ws),
            ConvGNGLU(base * 4, base * 4))

        # ── up path ────────────────────────────────────────────────────────
        self.up2  = Up(base * 4, base * 2)                    # ↑2  (B,192)
        self.up1  = Up(base * 2, base)                        # ↑2  (B, 96)

        # ── output conv ───────────────────────────────────────────────────
        self.out  = nn.Conv3d(base, ch, 3, 1, 1)
        nn.init.kaiming_uniform_(self.out.weight)

    def forward(self, x):
        x0 = self.enc0(x)                 # full-res
        x1 = self.attn1(self.down1(x0))   # ↓2 + attn
        x2 = self.bott(self.down2(x1))    # ↓4 + deep attn

        y1 = self.up2(x2) + x1            # merge level-2
        y0 = self.up1(y1)  + x0           # merge level-1

        return self.out(y0)

# ----------------------------------------------------------------
# cosine DDPM schedule
# ----------------------------------------------------------------
class CosineSched:
    def __init__(self, T=1000, s=.008):
        t = torch.arange(T + 1)
        abar = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
        abar = abar / abar[0]
        betas = 1 - abar[1:] / abar[:-1]
        self.betas = betas.float()
        self.alphas = 1 - self.betas
        self.abar = self.alphas.cumprod(0)
    def to(self, d):
        self.betas, self.alphas, self.abar = [
            x.to(d) for x in (self.betas, self.alphas, self.abar)]
        return self

SCHED = CosineSched()

def p_step(xt, eps, t):
    SCHED.to(xt.device)
    """DDPM posterior mean step (deterministic)."""
    b = SCHED.betas[t].view(-1, 1, 1, 1, 1)
    a = 1 - b
    abar = SCHED.abar[t].view(-1, 1, 1, 1, 1)
    return (1 / a.sqrt()) * (xt - b / (1 - abar).sqrt() * eps)

# ----------------------------------------------------------------
# Denoiser: adds a FiLM-style bias from timestep embedding
# ----------------------------------------------------------------
class Denoiser(nn.Module):
    def __init__(self, backbone: AttnUNet3D):
        super().__init__()
        self.bb = backbone
        self.proj = nn.Linear(128, 29)

    def forward(self, zt, t):
        B, _, D, H, W = zt.shape
        bias = self.proj(t_embed(t, 128)).view(B, 29, 1, 1, 1).expand(-1, -1, D, H, W)
        return self.bb(zt + bias)           # additive FiLM; still 29 ch

# ----------------------------------------------------------------
# Full model
# ----------------------------------------------------------------
class AttnUNetPartialDiff(nn.Module):
    """Super-resolves a 29-channel SH volume via attention U-Net + 4-step DDPM."""
    def __init__(self, steps=4, base=96, ws=8):
        super().__init__()
        self.steps = steps
        self.bb    = AttnUNet3D(29, base, ws)
        #self.den   = Denoiser(self.bb)

    def forward(self, x, return_sr0=False):
        sr0 = self.bb(x)          # deterministic SR
        #return x + sr0
        z = sr0
        for s in range(self.steps):
            t = torch.full((x.size(0),), self.steps - s - 1,
                           device=x.device, dtype=torch.long)
            eps = self.den(z, t)
            z = p_step(z, eps, t)
        return (z, sr0) if return_sr0 else z

# ----------------------------------------------------------------
if __name__ == "__main__":
    B, D, H, W = 2, 64, 64, 64
    inp = torch.randn(B, 29, D, H, W)
    mdl = AttnUNetPartialDiff().cuda()
    out = mdl(inp.cuda())
    print("output", out.shape)

