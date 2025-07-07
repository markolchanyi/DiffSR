import torch, torch.nn as nn, torch.nn.functional as F
from typing import Sequence

################################################################################
# Dual‑Branch SH Mixer — *stat‑free* version                                    #
# • No pre‑computed μ/σ.                                                         #
# • InstanceNorm3d (affine=True) handles per‑sample scaling.                     #
# • Low‑b + l0 channels go through an in‑network log1p/expm1 transform to tame   #
#   their positive dynamic range; the angular SH channels pass through untouched.#
################################################################################

class LogTransform(nn.Module):
    """Apply log1p to pos‑only channels 0‑1; inverse‑log on the way back."""
    def __init__(self):
        super().__init__()
    def forward(self, x, inverse: bool = False):
        if inverse:
            x[:, :2] = torch.expm1(x[:, :2])
        else:
            x[:, :2] = torch.log1p(x[:, :2].clamp(min=0))
        return x

class ChannelAffine(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, C, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, C, 1, 1, 1))
    def forward(self, x):
        return x * self.g + self.b

######################## Axial MLP‑Mixer block (unchanged) #####################
class AxialMLPBlock(nn.Module):
    def __init__(self, C, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        self.fc_d = nn.Linear(C, hidden)
        self.fc_h = nn.Linear(C, hidden)
        self.fc_w = nn.Linear(C, hidden)
        self.proj = nn.Linear(hidden, C)
        self.act  = nn.GELU()
    def _mix(self, x, axis, fc):
        B,C,D,H,W = x.shape
        if axis==0: y=x.permute(0,3,4,2,1).reshape(-1,D,C)
        if axis==1: y=x.permute(0,2,4,3,1).reshape(-1,H,C)
        if axis==2: y=x.permute(0,2,3,4,1).reshape(-1,W,C)
        y = self.proj(self.act(fc(self.norm(y))))
        if axis==0: y=y.view(B,H,W,D,C).permute(0,4,3,1,2)
        if axis==1: y=y.view(B,D,W,H,C).permute(0,4,1,3,2)
        if axis==2: y=y.view(B,D,H,W,C).permute(0,4,1,2,3)
        return y
    def forward(self,x):
        y=self._mix(x,0,self.fc_d)
        y=self._mix(y,1,self.fc_h)
        y=self._mix(y,2,self.fc_w)
        return x+y

###################### Spatial branch (unchanged, uses IN) ####################
class SpatialBranch(nn.Module):
    def __init__(self, C=29, base=96, hidden=192, depth=3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(C, base, 3,1,1), nn.InstanceNorm3d(base, affine=True), nn.GELU(),
            nn.Conv3d(base, base,3,1,1), nn.InstanceNorm3d(base, affine=True), nn.GELU())
        self.down = nn.Conv3d(base, base*2, 2,2)
        self.mlp  = nn.Sequential(*[AxialMLPBlock(base*2, hidden) for _ in range(depth)])
        self.up   = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False),
                                  nn.Conv3d(base*2, base,3,1,1))
        self.refine = nn.Sequential(nn.InstanceNorm3d(base,affine=True), nn.GELU(),
                                    nn.Conv3d(base, base,3,1,1), nn.InstanceNorm3d(base,affine=True), nn.GELU())
        self.out = nn.Conv3d(base, C,3,1,1)
        nn.init.zeros_(self.out.weight)
    def forward(self,x):
        x0=self.enc(x)
        x1=self.mlp(self.down(x0))
        x2=self.up(x1)+x0
        return self.out(self.refine(x2))

######################## SH per‑voxel Mixer ###################################
class SHMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_scalar = nn.Sequential(nn.Conv3d(2,8,1), nn.GELU(), nn.Conv3d(8,2,1))
        self.mlp_ang    = nn.Sequential(nn.Conv3d(27,128,1), nn.GELU(), nn.Conv3d(128,27,1))
        nn.init.zeros_(self.mlp_scalar[-1].weight)
        nn.init.zeros_(self.mlp_ang[-1].weight)
    def forward(self,x):
        s,a = x[:,:2], x[:,2:]
        return torch.cat([self.mlp_scalar(s), self.mlp_ang(a)],1)

######################## Full network #########################################
class DualBranchSHMixerSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.logtf = LogTransform()
        self.aff   = ChannelAffine(29)
        self.spatial = SpatialBranch()
        self.shmix   = SHMixer()

    def forward(self, x: torch.Tensor):   # x low‑res, B,C,D,H,W
        #up = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        up = x
        y  = self.logtf(up, inverse=False)      # tame dynamic range
        y  = self.aff(y)
        z  = y + self.spatial(y) + self.shmix(y)  # residual per branch
        out = self.logtf(z, inverse=True)       # back to native units
        return out

