# s2_unet_global.py — memory‑light version (no external deps)

from __future__ import annotations
import torch, torch.nn as nn
from functools import lru_cache
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import Linear, FullyConnectedTensorProduct


# Icosahedral grid helper
try:
    from e3nn.util.grid import icosahedral_sphere
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


# MRtrix to canonical sign mask. This needs to be set since the o3nn libraries 
# are super specific wrt. to SH ordering :)
_sign = torch.tensor([1,1]+[(-1)**m for l in (2,4,6) for m in range(-l,l+1)],dtype=torch.float32)  # (29,)

# Projection matrix from SH to isohedral amplitudes (careful with number!!!)
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
    return torch.cat(Ys,1) # (N,27)




# ------ helpers ----------
def irrep_list(k: int):
    scalars = ["0e"]                     # keep only ONE SiLU scalar
    gates   = ["0e"] * (3 * k)           # one gate scalar per gated irrep
    gated   = (["2e", "4e", "6e"] * k)   # the vectors/tensors to gate
    return scalars, gates, gated


class IrrepMLPBlock(nn.Module):
    def __init__(self, k=2, use_tp=True):
        super().__init__()

        # I/O irreps are always the base 28-dim set
        ir_io = o3.Irreps("0e + 2e + 4e + 6e")

        scalars, gates, gated = irrep_list(k)

        ir_scalar = o3.Irreps(" + ".join(scalars))   # 1 × 0e
        ir_gates  = o3.Irreps(" + ".join(gates))     # 3k × 0e
        ir_gated  = o3.Irreps(" + ".join(gated))     # 3k irreps

        ir_pre    = ir_scalar + ir_gates + ir_gated  # before Gate

        self.lin1 = Linear(ir_io, ir_pre)
        self.gate = Gate(
            irreps_scalars = ir_scalar,
            act_scalars = [nn.SiLU()],                # length 1
            irreps_gates = ir_gates,
            act_gates = [nn.Sigmoid()] * ir_gates.num_irreps,
            irreps_gated = ir_gated
        )
        self.mix  = (FullyConnectedTensorProduct(ir_scalar + ir_gated,
                                                 ir_scalar + ir_gated,
                                                 ir_scalar + ir_gated) if use_tp else nn.Identity())

        self.lin2 = Linear(ir_scalar + ir_gated, ir_io)

    def forward(self, x):                # (..., 28)
        x = self.lin1(x)
        x = self.gate(x)

        # call the tensor-product with BOTH arguments
        if isinstance(self.mix, FullyConnectedTensorProduct):
            x = self.mix(x, x)           # use the same tensor twice
        else:
            x = self.mix(x)              # Identity()

        x = self.lin2(x)
        return x


class IrrepMLP(nn.Module):
    def __init__(self, depth=5, k=3, use_tp=True):
        super().__init__()
        self.blocks = nn.ModuleList([IrrepMLPBlock(k, use_tp) for _ in range(depth)])
        self.g      = nn.Parameter(torch.zeros(depth))     # ReZero gains

    def forward(self, c):                      # c (..., 29)
        b0   = c[..., :1]
        iso  = c[..., 1:2]
        sh27 = torch.cat([c[..., 2:7], c[..., 7:16], c[..., 16:]], -1)
        x = torch.cat([iso, sh27], -1)         # (..., 28)

        for gain, blk in zip(self.g, self.blocks):
            x = x + gain * blk(x)              # learnable residual

        iso_o, sh27_o = x[..., :1], x[..., 1:]
        return torch.cat([b0, iso_o, sh27_o], -1)



class VoxelS2(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = IrrepMLP(depth=5, k=2, use_tp=False)

    def forward(self, x):
        B,C,D,H,W = x.shape
        x = x.permute(0,2,3,4,1).reshape(-1,C)
        x = self.core(x)
        return x.view(B,D,H,W,C).permute(0,4,1,2,3).contiguous()




## tokenize stuff to separate out non-SH from SH
class GTok(nn.Module):
    def __init__(self, ch, nt=8):
        super().__init__()
        self.tok  = nn.Parameter(torch.randn(1, nt, ch)) #(1, T, C)
        self.attn = nn.MultiheadAttention(ch, 4, batch_first=True)
        self.ln   = nn.LayerNorm(ch)

    def forward(self, x):  # x: (B,C,D,H,W)
        B, C, D, H, W = x.shape
        seq = x.flatten(2).permute(0, 2, 1)   # (B, N, C)
        tok = self.tok.expand(B, -1, -1)      # (B, T, C)
        tok, _ = self.attn(tok, seq, seq)
        g = self.ln(tok.mean(dim=1))
        g = g.view(B, C, 1, 1, 1)             # broadcast incase
        return x + g



class CBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(cin,cout,3,padding=1),nn.GELU(),nn.Conv3d(cout,cout,3,padding=1),nn.GELU())
    def forward(self,x): return self.net(x)



class UNet3d(nn.Module):
    def __init__(self,in_ch=29,base=192,depth=3):
        super().__init__()
        self.enc=nn.ModuleList()
        self.pool=nn.ModuleList()
        self.dec=nn.ModuleList()
        ch=in_ch

        for d in range(depth):
            out=base*2**d
            self.enc.append(CBlock(ch,out))
            ch=out
            self.pool.append(nn.AvgPool3d(2)) if d<depth-1 else None  # avpool for now since it works

        self.gtok=GTok(ch)

        for d in reversed(range(depth-1)):
            out=base*2**d
            # transposed convs sometimes and sometimes dont give checkerboard artifacts for the HCP data, but more often dont so keep
            self.dec.append(nn.Sequential(nn.ConvTranspose3d(ch,out,2,2),nn.GELU(),CBlock(out,out)))
            ch=out
        self.final=nn.Conv3d(ch,in_ch,1)
    def forward(self,x):
        skips=[]; h=x
        for enc,pool in zip(self.enc[:-1],self.pool):
            h=enc(h)
            skips.append(h)
            h=pool(h)

        h=self.enc[-1](h); h=self.gtok(h)

        for dec in self.dec:
            h=dec(h)
            h=h+skips.pop()

        return self.final(h)




class S2UNetGlobal(nn.Module):
    def __init__(self, base=256):
        super().__init__()
        self.register_buffer("sign",_sign.view(1,29,1,1,1))
        self.enc=VoxelS2()
        self.unet=UNet3d(29,base)
        self.dec=VoxelS2()

    def forward(self,x):
        x=x*self.sign
        x=self.dec(self.unet(self.enc(x)))

        return x*self.sign




def _smoke():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    x=torch.randn(3,29,64,64,64,device=dev)
    y=S2UNetGlobal().to(dev)(x)
    torch.cuda.synchronize()         # make sure all kernels are done

    current  = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    peak     = torch.cuda.max_memory_allocated(dev)

    print(f"Current  GPU mem  : {current/2**20:7.2f} MB")
    print(f"Reserved GPU mem  : {reserved/2**20:7.2f} MB")
    print(f"Peak     GPU mem  : {peak/2**20:7.2f} MB")
    print(sum(p.numel() for p in S2UNetGlobal().parameters()) / 1e6, "M params")
    print(" shapes:", x.shape, "→", y.shape)

if __name__=="__main__": _smoke()

