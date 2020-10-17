import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict


class Mean(nn.Module):
  def __init__(self, dim: list, keepdim=False):
    super().__init__()
    self.dim = dim
    self.keepdim = keepdim

  def forward(self, x):
    return torch.mean(x, self.dim, self.keepdim)


class PretrainNet(pl.LightningModule):
  def train(self, mode: bool):
    return super().train(False)

  def state_dict(self, destination, prefix, keep_vars):
    destination = OrderedDict()
    destination._metadata = OrderedDict()
    return destination

  def setup(self, device: torch.device):
    self.freeze()


