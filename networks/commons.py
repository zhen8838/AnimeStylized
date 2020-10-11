import torch
import torch.nn as nn
import torch.nn.functional as F


class Mean(nn.Module):
  def __init__(self, dim: list, keepdim=False):
    super().__init__()
    self.dim = dim
    self.keepdim = keepdim

  def forward(self, x):
    return torch.mean(x, self.dim, self.keepdim)
