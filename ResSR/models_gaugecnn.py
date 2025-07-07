# gauge_cnn_sh_super_res.py
"""Gauge‑equivariant CNN for super‑resolving diffusion‑MRI SH volumes.

Channel layout (MRtrix‑style, l≤6)
----------------------------------
0. **mean‑b0**      – scalar, *not* an SH coefficient but must be SR‑ed
1. l=0, m=0         – scalar (0‑order SH)
2‑6.   l=2, m=‑2…2  – 5‑vector
7‑15.  l=4, m=‑4…4  – 9‑vector
16‑28. l=6, m=‑6…6  – 13‑vector

Total = 29 channels → irreps = `1x0e (b0) + 1x0e (l0) + 2e + 4e + 6e` which
in e3nn shorthand is **`2x0e + 2e + 4e + 6e`**.

The network is SE(3)‑equivariant: rotations of the input volume lead to the
corresponding rotated output, preventing the "upsample blur" that arises when
angular information is averaged out in vanilla 3‑D ConvNets.

Requires: e3nn ≥ 0.7, torch_cluster, torch_scatter, torch_sparse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from e3nn import o3
from e3nn.nn.models.gate_points_2102 import Network as GatePointsNetwork


#try:
#    from e3nn import o3
#    from e3nn.nn.models.v2106 import SimpleBlockSeries
#    from torch_cluster import radius_graph
#except ImportError as err:
#    raise ImportError(
#        "Missing equivariant deps. Install with:\n"
#        "  pip install e3nn torch_cluster torch_scatter torch_sparse") from err


class SHGaugeSRNet(nn.Module):
    """SE(3)‑equivariant super‑resolution network for 29‑channel SH volumes."""

    def __init__(
        self,
        hidden_irreps: str = "32x0e + 16x2e + 16x4e + 8x6e",
        layers: int = 8,
        radius: float = 2.5,
        neighbors: int = 24,
        upsample_factor: int = 2,
    ) -> None:
        super().__init__()

        # Two scalars: mean‑b0 + l=0, plus l=2,4,6 bands.
        self.irreps_in = o3.Irreps("2x0e + 2e + 4e + 6e")
        self.irreps_out = self.irreps_in
        self.radius = radius
        self.neighbors = neighbors
        self.upsample_factor = upsample_factor

        # Backbone: 8 stacked equivariant blocks with learnable radial MLPs.
        self.backbone = SimpleBlockSeries(
            irreps_in=self.irreps_in,
            irreps_hidden=hidden_irreps,
            irreps_out=self.irreps_out,
            layers=layers,
            radial_layers=2,
            radius=radius,
            num_neighbors=neighbors,
        )

    # ------------------------------------------------------------------
    # utility functions
    # ------------------------------------------------------------------
    @staticmethod
    def _grid(D: int, H: int, W: int, device, dtype) -> torch.Tensor:
        """Regular 3‑D voxel grid → (N,3) xyz coordinates in *voxel units*."""
        z, y, x = torch.meshgrid(
            torch.arange(D, device=device, dtype=dtype),
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((x, y, z), dim=-1).reshape(-1, 3)

    def _graph(self, pos: torch.Tensor) -> torch.Tensor:
        """Build a radius graph once per volume shape."""
        return radius_graph(
            pos,
            r=self.radius,
            loop=False,
            max_num_neighbors=self.neighbors,
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super‑resolve a 29‑channel SH volume.

        Parameters
        ----------
        x : (B, 29, D, H, W) – low‑res input

        Returns
        -------
        (B, 29, D*F, H*F, W*F) – high‑res output, F = upsample_factor
        """
        B, C, D, H, W = x.shape
        if C != 29:
            raise ValueError(f"Expected 29 channels, got {C}.")

        # naive trilinear upsampling to target grid ----------------------
        if self.upsample_factor > 1:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_factor,
                mode="trilinear",
                align_corners=False,
            )
            D, H, W = x.shape[2:]

        # build voxel graph (shared across batch) ------------------------
        pos = self._grid(D, H, W, x.device, x.dtype)
        edge_index = self._graph(pos)
        N = pos.shape[0]

        # flatten spatial dims into graph nodes --------------------------
        feats = x.permute(0, 2, 3, 4, 1).reshape(B * N, C)  # (B·N,29)

        # equivariant processing ----------------------------------------
        out = self.backbone(feats, pos.repeat(B, 1), edge_index)

        # reshape back to 5‑D tensor ------------------------------------
        out = (
            out.view(B, D, H, W, C)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        return out


# ----------------------------------------------------------------------
# quick sanity check ----------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    model = SHGaugeSRNet()
    low_res = torch.randn(1, 29, 24, 24, 24)
    high_res = model(low_res)
    print("Output shape:", high_res.shape)

