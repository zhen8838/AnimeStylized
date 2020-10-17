import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class AnimeDiscriminator(nn.Module):
  def __init__(self, channel: int = 64, nblocks: int = 3) -> None:
    super().__init__()
    channel = channel // 2
    last_channel = channel
    f = [
        spectral_norm(nn.Conv2d(3, channel, 3, stride=1, padding=1, bias=False)),
        nn.LeakyReLU(0.2, inplace=True)
    ]
    in_h = 256
    for i in range(1, nblocks):
      f.extend([
          spectral_norm(nn.Conv2d(last_channel, channel * 2,
                                  3, stride=2, padding=1, bias=False)),
          nn.LeakyReLU(0.2, inplace=True),
          spectral_norm(nn.Conv2d(channel * 2, channel * 4,
                                  3, stride=1, padding=1, bias=False)),
          nn.InstanceNorm2d(channel * 4, affine=True),
          nn.LeakyReLU(0.2, inplace=True)
      ])
      last_channel = channel * 4
      channel = channel * 2
      in_h = in_h // 2

    self.body = nn.Sequential(*f)

    self.head = nn.Sequential(*[
        spectral_norm(nn.Conv2d(last_channel, channel * 2, 3,
                                stride=1, padding=1, bias=False)),
        nn.InstanceNorm2d(channel * 2, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(channel * 2, 1, 3, stride=1, padding=1, bias=False))])

  def forward(self, x):
    x = self.body(x)
    x = self.head(x)
    return x


class Conv2DNormLReLU(nn.Module):
  def __init__(self,
               in_channels: int,
               out_channels: int,
               kernel_size: int = 3,
               stride: int = 1,
               padding: int = 1,
               bias=False) -> None:
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels,
                          kernel_size, stride,
                          padding, bias=bias)
    self.norm = nn.InstanceNorm2d(out_channels,
                                  affine=True)
    self.lrelu = nn.LeakyReLU(0.2, inplace=True)

  def forward(self, x):
    x = self.conv(x)
    x = self.norm(x)
    x = self.lrelu(x)
    return x


class resBlock(nn.Module):
  def __init__(self,
               in_channels: int,
               out_channels: int) -> None:
    super().__init__()
    self.body = nn.Sequential(
        Conv2DNormLReLU(in_channels, out_channels, 1, padding=0),
        Conv2DNormLReLU(out_channels, out_channels, 3),
        nn.Conv2d(out_channels, out_channels // 2, 1, bias=False)
    )

  def forward(self, x0):
    x = self.body(x0)
    return x0 + x


class AnimeGeneratorLite(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.A = nn.Sequential(
        Conv2DNormLReLU(3, 32, 7, padding=3),
        Conv2DNormLReLU(32, 32, stride=2),
        Conv2DNormLReLU(32, 32))

    self.B = nn.Sequential(
        Conv2DNormLReLU(32, 64, stride=2),
        Conv2DNormLReLU(64, 64),
        Conv2DNormLReLU(64, 64))

    self.C = nn.Sequential(
        resBlock(64, 128),
        resBlock(64, 128),
        resBlock(64, 128),
        resBlock(64, 128))

    self.D = nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=2),
        Conv2DNormLReLU(64, 64),
        Conv2DNormLReLU(64, 64),
        Conv2DNormLReLU(64, 64))

    self.E = nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=2),
        Conv2DNormLReLU(64, 32),
        Conv2DNormLReLU(32, 32),
        Conv2DNormLReLU(32, 32, 7, padding=3))

    self.out = nn.Sequential(
        nn.Conv2d(32, 3, 1, bias=False),
        nn.Tanh())

  def forward(self, x):
    x = self.A(x)
    x = self.B(x)
    x = self.C(x)
    x = self.D(x)
    x = self.E(x)
    x = self.out(x)
    return x
