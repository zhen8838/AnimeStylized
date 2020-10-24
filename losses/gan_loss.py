import torch
import torch.nn as nn
import torch.nn.functional as F


class GanLoss(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, real_logit, fake_logit):
    real_logit = F.sigmoid(real_logit)
    fake_logit = F.sigmoid(fake_logit)

    g_loss_blur = -torch.mean(torch.log(fake_logit))
    d_loss_blur = -torch.mean(torch.log(real_logit) + torch.log(1. - fake_logit))

    return d_loss_blur, g_loss_blur


class LSGanLoss(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    # NOTE c=b a=0

  def _d_loss(self, real_logit, fake_logit):
    # 1/2 * [(real-b)^2 + (fake-a)^2]
    return 0.5 * (torch.mean((real_logit - 1)**2) + torch.mean(fake_logit**2))

  def _g_loss(self, fake_logit):
    # 1/2 * (fake-c)^2
    return torch.mean((fake_logit - 1)**2)

  def forward(self, real_logit, fake_logit):
    g_loss = self._g_loss(fake_logit)
    d_loss = self._d_loss(real_logit, fake_logit)
    return d_loss, g_loss
