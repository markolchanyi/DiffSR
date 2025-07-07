import torch, e3nn
from e3nn.o3 import Irreps, Linear
print("e3nn version:", e3nn.__version__)

lin = Linear(Irreps("644x0e"), Irreps("64x0e"))  # traced *immediately* (CPU)
x = torch.randn(10, 644, device="cuda")          # GPU tensor
lin.to("cuda")(x)                                # crashes with reshape 644
