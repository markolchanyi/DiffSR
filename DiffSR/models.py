"""
3-D UNet with S2 graph-conv flanks
"""

from __future__ import annotations
import torch
import torch.nn as nn
from functools import lru_cache
from e3nn import o3
import numpy as np


# Icosahedral sphere + kNN graph
#@TODO
try:
    from e3nn.util.grid import icosahedral_sphere as _e3nn_icosphere  # DEBUG...sometimes gets buggy in new e3nn
except Exception:
    _e3nn_icosphere = None


def icosahedral_sphere(level=1):

    if _e3nn_icosphere is not None:
        verts = _e3nn_icosphere(level)
        if isinstance(verts, torch.Tensor):
            verts = verts.detach().cpu().numpy()
        return verts

    # sometimes doesn't work idk
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    verts = np.array(
        [[-1,  phi, 0],
            [ 1,  phi, 0],
            [-1, -phi, 0],
            [ 1, -phi, 0],
            [0, -1,  phi],
            [0,  1,  phi],
            [0, -1, -phi],
            [0,  1, -phi],
            [ phi, 0, -1],
            [ phi, 0,  1],
            [-phi, 0, -1],
            [-phi, 0,  1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array(
        [[0, 11, 5], [0, 5, 1],  [0, 1, 7],  [0, 7, 10], [0, 10, 11],
            [1, 5, 9],  [5, 11, 4], [11, 10, 2],[10, 7, 6], [7, 1, 8],
            [3, 9, 4],  [3, 4, 2],  [3, 2, 6],  [3, 6, 8],  [3, 8, 9],
            [4, 9, 5],  [2, 4, 11], [6, 2, 10], [8, 6, 7],  [9, 8, 1],
        ],
        dtype=np.int64,
    )

    for _ in range(level):
        vlist = verts.tolist()
        flist = []
        mid = {}

        def midpoint(a: int, b: int):
            key = tuple(sorted((a, b)))
            if key in mid:
                return mid[key]
            v = (verts[a] + verts[b]) * 0.5
            v /= np.linalg.norm(v)
            mid[key] = len(vlist)
            vlist.append(v)
            return mid[key]

        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            flist += [[a, ab, ca],
                [b, bc, ab],
                [c, ca, bc],
                [ab, bc, ca],
            ]

        verts = np.asarray(vlist, dtype=np.float64)
        faces = np.asarray(flist, dtype=np.int64)

    return verts


def build_knn_graph(verts, k=6):
    N = verts.shape[0]
    diff = verts[:, None, :] - verts[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)      # chord distances
    np.fill_diagonal(d2, np.inf)           # exclude self

    neighbors = np.argpartition(d2, kth=k, axis=1)[:, :k]
    return neighbors.astype(np.int64)


# SH amplitude to ico grids

@lru_cache(maxsize=4) #just turn off lru if theres trouble
def _sh_basis_l2(level: int):
    verts = icosahedral_sphere(level)
    dirs = torch.tensor(verts, dtype=torch.float32)
    try:
        Y = o3.spherical_harmonics(2, dirs, normalization="component")
    except TypeError:  # older e3nn
        Y = o3.spherical_harmonics(2, dirs, normalize=True)
    return Y


@lru_cache(maxsize=4) #same

# pseudoinverse through pinv
def _sh_pinv_l2(level):
    Y = _sh_basis_l2(level)         # (N, 5)
    Y_pinv = torch.linalg.pinv(Y)
    return Y_pinv


# Ico graph-conv layers

class IcoGraphConvLayer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        neighbors: torch.LongTensor,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.register_buffer("neighbors", neighbors, persistent=False)  # (42, K)

        self.lin = nn.Linear(in_dim, out_dim)
        self.self_weight = nn.Parameter(torch.tensor(1.0))
        self.neigh_weight = nn.Parameter(torch.tensor(1.0))
        self.ln = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, N, F_in = x.shape
        assert F_in == self.in_dim, f"Expected F_in={self.in_dim}, got {F_in}"

        h = self.lin(x)

        # get neighbors back
        nb = self.neighbors
        h_nb = h[:, nb, :]
        h_neigh = h_nb.mean(dim=2)

        out = self.self_weight * h + self.neigh_weight * h_neigh
        out = self.ln(out)
        out = self.act(out)
        return out


# Takes SH coeffs at each voxel, projects l2 to amps, then graph-conv layers, then back to amplitudes via pseudoinserse
class S2IcoGraphBlockL2(nn.Module):

    def __init__(
        self,
        level=1,
        k=6,
        feat_dim=16,
        n_layers=2,
    ):
        super().__init__()
        verts = icosahedral_sphere(level)
        neighbors_np = build_knn_graph(verts, k=k)
        neighbors = torch.from_numpy(neighbors_np).long()

        Y_l2 = _sh_basis_l2(level)
        Y_pinv = _sh_pinv_l2(level)

        self.level = level
        self.register_buffer("Y_l2", Y_l2, persistent=False) #funky
        self.register_buffer("Y_l2_pinv", Y_pinv, persistent=False)

        gc_layers = []
        in_dim = 1  # scaler amplitude per direction
        for _ in range(n_layers):
            gc_layers.append(IcoGraphConvLayer(in_dim, feat_dim, neighbors))
            in_dim = feat_dim
        self.gc_layers = nn.ModuleList(gc_layers)
        self.out_linear = nn.Linear(in_dim, 1)

    def forward(self, c):
        *spatial, C = c.shape
        assert C == 7, f"S2IcoGraphBlockL2 expects 7 channels, got {C}"
        # ^ b0 + l0 + l2

        x = c.reshape(-1, C)  # where  prod(spatial)

        b0 = x[:, :1]
        l0 = x[:, 1:2]
        c_l2 = x[:, 2:]

        # ampitudes on ico grid
        Y_l2 = self.Y_l2
        amp = c_l2 @ Y_l2.T

        # graph conv on amps(?)
        h = amp.unsqueeze(-1)
        for gc in self.gc_layers:
            h = gc(h)

        amp_out = self.out_linear(h).squeeze(-1)

        # amplitudes -> SH (l=2) via pseudo-inverse
        Y_pinv = self.Y_l2_pinv
        c_l2_out = amp_out @ Y_pinv.T

        out = torch.cat([b0, l0, c_l2_out], dim=-1)
        return out.view(*spatial, C)


class VoxelS2GraphL2(nn.Module):

    def __init__(self, **s2_kwargs):
        super().__init__()
        self.core = S2IcoGraphBlockL2(**s2_kwargs)

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        assert C == 7, f"VoxelS2GraphL2 expects 7 channels, got {C}"
        # move channels to last dim then move back
        x_spatial_last = x.permute(0, 2, 3, 4, 1).contiguous()
        x_proc = self.core(x_spatial_last)
        return x_proc.permute(0, 4, 1, 2, 3).contiguous()


# UNet backbone

# Global token block
class GTok(nn.Module):

    def __init__(self, ch: int, nt: int = 8):
        super().__init__()
        self.tok = nn.Parameter(torch.randn(1, nt, ch))
        self.attn = nn.MultiheadAttention(ch, 4, batch_first=True)
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        # x: (B,C,D,H,W)
        B, C, D, H, W = x.shape
        seq = x.flatten(2).permute(0, 2, 1)
        tok = self.tok.expand(B, -1, -1)
        tok, _ = self.attn(tok, seq, seq)     # global tokens attend to volume
        g = self.ln(tok.mean(dim=1))
        g = g.view(B, C, 1, 1, 1)
        return x + g


class CBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet3d(nn.Module):
    def __init__(self, in_ch=7, base=192, depth=4):
        super().__init__()
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.dec = nn.ModuleList()
        ch = in_ch

        # encoder
        for d in range(depth):
            out = base * 2 ** d
            self.enc.append(CBlock(ch, out))
            ch = out
            if d < depth - 1:
                self.pool.append(nn.AvgPool3d(2))

        # then global token block
        self.gtok = GTok(ch)

        # decoder
        for d in reversed(range(depth - 1)):
            out = base * 2 ** d
            self.dec.append(
                nn.Sequential(
                    nn.ConvTranspose3d(ch, out, 2, 2),
                    nn.GELU(),
                    CBlock(out, out),
                )
            )
            ch = out

        self.final = nn.Conv3d(ch, in_ch, 1)

    def forward(self, x):
        skips = []
        h = x

        # encoder
        for enc, pool in zip(self.enc[:-1], self.pool):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        # bottleneck
        h = self.enc[-1](h)
        h = self.gtok(h)

        # decoder
        for dec in self.dec:
            h = dec(h)
            h = h + skips.pop()

        return self.final(h)


# Sign pattern for MRtrix real SH basis up to l=2
def _make_sign_l2():
    signs = [1.0, 1.0]  # b0, l0
    signs += [float((-1) ** m) for m in range(-2, 3)]
    return torch.tensor(signs, dtype=torch.float32)


# FULL ARCHITECTURE
# inp / out: (B, 7, x, y, z)
class S2UNetGlobalL2(nn.Module):

    def __init__(
        self,
        base=128,
        level=1,
        k=6,
        feat_dim=16,
        n_layers=2,
    ):
        super().__init__()
        sign = _make_sign_l2().view(1, 7, 1, 1, 1)
        self.register_buffer("sign", sign)

        self.enc = VoxelS2GraphL2(level=level, k=k, feat_dim=feat_dim, n_layers=n_layers)
        self.unet = UNet3d(in_ch=7, base=base)
        self.dec = VoxelS2GraphL2(level=level, k=k, feat_dim=feat_dim, n_layers=n_layers)

    def forward(self, x: torch.Tensor):
        # enforce antipodal sign convention
        x = x * self.sign
        x = self.enc(x)
        x = self.unet(x)
        x = self.dec(x)
        return x * self.sign


