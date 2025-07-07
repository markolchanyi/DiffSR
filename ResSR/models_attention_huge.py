"""
Residual‑in‑Residual Dense UNet (RRD‑UNet)  – **no global skip**
───────────────────────────────────────────────────────────────────────────────
* Spatial branch: 2‑level UNet; every block is an RRDB.
* Order‑wise 1×1 SH mixer.
* No α‑scaled identity path – the network learns the full mapping.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Utility layers
# ──────────────────────────────────────────────────────────────────────────────
class LogTransform(nn.Module):
    """Log1p / Exp1m on channels 0–1 (low‑b + ℓ₀)."""
    def forward(self, x: torch.Tensor, inverse: bool = False):
        if inverse:
            x[:, :2] = torch.expm1(x[:, :2])
        else:
            x[:, :2] = torch.log1p(x[:, :2].clamp_min_(0))
        return x

class ChannelAffine(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, C, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, C, 1, 1, 1))
    def forward(self, x):
        return x * self.g + self.b

# ──────────────────────────────────────────────────────────────────────────────
# RRDB components
# ──────────────────────────────────────────────────────────────────────────────
class DenseConv(nn.Module):
    def __init__(self, C: int, growth: int = 32):
        super().__init__()
        self.conv = nn.Conv3d(C, growth, 3, 1, 1)
        nn.init.kaiming_uniform_(self.conv.weight, a=0.2)
        self.act  = nn.GELU()
    def forward(self, x):
        y = self.act(self.conv(x))
        return torch.cat([x, y], 1)

class ResidualDenseBlock(nn.Module):
    def __init__(self, Cin: int, growth: int = 32):
        super().__init__()
        layers, Ccur = [], Cin
        for _ in range(5):
            layers.append(DenseConv(Ccur, growth))
            Ccur += growth
        self.dense = nn.Sequential(*layers)
        self.fuse  = nn.Conv3d(Ccur, Cin, 1)
        nn.init.kaiming_uniform_(self.fuse.weight, a=0.2)
    def forward(self, x):
        return x + self.fuse(self.dense(x))   # no extra scaling

class RRDB(nn.Module):
    def __init__(self, C: int, growth: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            ResidualDenseBlock(C, growth),
            ResidualDenseBlock(C, growth),
            ResidualDenseBlock(C, growth))
    def forward(self, x):
        return x + self.body(x)               # no extra scaling

# ──────────────────────────────────────────────────────────────────────────────
# RRD‑UNet spatial branch
# ──────────────────────────────────────────────────────────────────────────────
class WindowAttn3D(nn.Module):
    """8×8×8 shifted-window self-attention (heads=4)."""
    def __init__(self, dim, heads=4, ws=8):
        super().__init__()
        self.ws, self.h, self.scale = ws, heads, (dim // heads) ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        size = 2*ws - 1
        self.rel = nn.Parameter(torch.zeros(heads, size, size, size))
        nn.init.trunc_normal_(self.rel, .02)

    def forward(self, x):
        B,C,D,H,W = x.shape; ws,h = self.ws, self.h
        s = ws // 2
        x = torch.roll(x, (-s,-s,-s), (2,3,4))
        x = (x.reshape(B,C,D//ws,ws,H//ws,ws,W//ws,ws)
               .permute(0,2,4,6,3,5,7,1)
               .reshape(-1, ws**3, C))
        q,k,v = self.qkv(x).reshape(-1, ws**3, 3, h, C//h).unbind(2)
        q,k,v = [t.permute(0,2,1,3) for t in (q,k,v)]
        attn  = (q * self.scale) @ k.transpose(-2,-1)
        coords = torch.stack(
            torch.meshgrid([torch.arange(ws)]*3, indexing='ij')).flatten(1)
        rel = coords[:, :, None] - coords[:, None, :] + ws - 1
        attn = attn + self.rel[:, rel[0], rel[1], rel[2]]
        attn = attn.softmax(-1)
        out  = (attn @ v).transpose(2,3).transpose(1,2)\
                .reshape(-1, ws**3, C)
        out  = self.proj(out).reshape(
                B, D//ws, H//ws, W//ws, ws, ws, ws, C)\
                .permute(0,7,1,4,2,5,3,6)\
                .reshape(B,C,D,H,W)
        return torch.roll(out, (s,s,s), (2,3,4))

# ─────────────────────────────────────────────────────────────
class RRDU(nn.Module):
    """
    RRDB-UNet with two scales + window attention bottleneck.
    Output: same spatial size, C channels.
    """
    def __init__(self, C=29, base=96):
        super().__init__()
        g1, g2 = 32, 40                 # growth for shallow / deep RRDBs

        self.enc0 = nn.Sequential(      # ① shallow
            nn.Conv3d(C, base, 3, 1, 1),
            nn.GroupNorm(8, base, affine=False),
            nn.GELU(),
            RRDB(base, growth=g1))

        self.down1 = nn.Conv3d(base, base*2, 2, 2)        # ↓2
        self.enc1  = RRDB(base*2, growth=g1)

        self.down2 = nn.Conv3d(base*2, base*4, 2, 2)      # ↓4
        self.bottl = nn.Sequential(
            RRDB(base*4, growth=g2),
            WindowAttn3D(base*4, heads=4, ws=8))

        # up path
        self.up2   = nn.Sequential(
            nn.ConvTranspose3d(base*4, base*2*8, 2, 2),   # 3-D pixel shuffle
            nn.PixelShuffle(2))                           # ↑2

        self.dec1  = RRDB(base*2, growth=g1)

        self.up1   = nn.Sequential(
            nn.ConvTranspose3d(base*2, base*8, 2, 2),
            nn.PixelShuffle(2))

        self.dec0  = RRDB(base, growth=g1)

        self.out   = nn.Conv3d(base, C, 3, 1, 1)
        nn.init.kaiming_uniform_(self.out.weight, a=0.2)

    def forward(self, x):
        x0 = self.enc0(x)               # (base)
        x1 = self.enc1(self.down1(x0))  # (base*2)
        x2 = self.bottl(self.down2(x1)) # (base*4)
        y1 = self.dec1(self.up2(x2) + x1)
        y0 = self.dec0(self.up1(y1) + x0)
        return self.out(y0)

# ──────────────────────────────────────────────────────────────────────────────
# Order‑wise SH mixer
# ──────────────────────────────────────────────────────────────────────────────
_SL: Dict[str, slice] = {
    'lb': slice(0, 1),
    'l0': slice(1, 2),
    'l2': slice(2, 7),
    'l4': slice(7, 16),
    'l6': slice(16, 29),
}
class SHMixer(nn.Module):
    def __init__(self, hidden=64, mix_lowb=False):
        super().__init__()
        def block(ch):
            return nn.Sequential(nn.Conv3d(ch, hidden, 1), nn.GELU(), nn.Conv3d(hidden, ch, 1))
        self.blocks = nn.ModuleDict({
            'lb': block(1) if mix_lowb else nn.Identity(),
            'l0': block(1),
            'l2': block(5),
            'l4': block(9),
            'l6': block(13)})
        for m in self.blocks.values():
            if isinstance(m, nn.Sequential):
                nn.init.kaiming_uniform_(m[-1].weight, a=0.2)
    def forward(self, x):
        comps = [self.blocks[k](x[:, sl]) for k, sl in _SL.items()]
        return torch.cat(comps, 1)

# ──────────────────────────────────────────────────────────────────────────────
# Full model (no global skip)
# ──────────────────────────────────────────────────────────────────────────────
class RRDUNetSHSR(nn.Module):
    def __init__(self, mix_lowb=False):
        super().__init__()
        self.log  = LogTransform()
        self.aff  = ChannelAffine(29)
        self.rrdu = RRDU()
        self.mix  = SHMixer(mix_lowb=mix_lowb)
    def forward(self, x: torch.Tensor):
        y = self.aff(self.log(x, inverse=False))
        z = self.rrdu(y) + self.mix(y)         # full mapping, no skip
        return self.log(z, inverse=True)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    net = RRDUNetSHSR()
    x   = torch.randn(1, 29, 64, 64, 64)
    out = net(x)
    print(out.shape)  #  (1, 29, 64, 64, 64)

