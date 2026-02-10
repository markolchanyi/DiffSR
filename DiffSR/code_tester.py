import torch
from torch_cluster import grid_cluster

pos = torch.tensor([[0., 0.], [11., 9.], [2., 8.], [2., 2.], [8., 3.]])
size = torch.Tensor([5, 5])

cluster = grid_cluster(pos, size)
print(cluster)
