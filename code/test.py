import torch
import torch.nn as nn
import torch.functional as F


B, C, T , W, H= 10, 20, 30, 40, 50
i = torch.rand(B, C, T, W, H)

net = nn.Conv3d(C, 30, kernel_size=3, padding='valid')

out = net(i)

print(i.shape)
print(out.shape)