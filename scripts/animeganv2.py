import os
import sys
sys.path.insert(0, os.getcwd())
import pytorch_lightning as pl
from networks.gan import SpectNormDiscriminator, UnetGenerator, VGGPreTrained, AnimeDiscriminator
from datasets.whiteboxgan import WhiteBoxGanDataModule, denormalize
from losses.gan_loss import LSGanLoss
from losses.function import variation_loss, rgb2yuv
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as nf
import torch.functional as F
import torchvision.transforms.functional as tf
from scripts.common import run_train, log_images
from scripts.animegan import AnimeGAN


class AnimeGANv2(AnimeGAN):

  def __init__(
      self,
      lr_g: float = 2e-4,
      lr_d: float = 2e-4,
      g_adv_weight: float = 300.,
      d_adv_weight: float = 300.,
      con_weight: float = 1.5,
      sty_weight: float = 2.8,
      color_weight: float = 10.,
      tv_weight: float = 1.,
      pre_trained_ckpt: str = None,
      **kwargs
  ):
    super().__init__()
    self.save_hyperparameters()

    # networks
    self.generator = UnetGenerator()
    if pre_trained_ckpt:
      ckpt = torch.load(pre_trained_ckpt)
      generatordict = dict(filter(lambda k: 'generator' in k[0], ckpt['state_dict'].items()))
      generatordict = {k.split('.', 1)[1]: v for k, v in generatordict.items()}
      self.generator.load_state_dict(generatordict, True)
      del ckpt
      del generatordict
      print("Success load pretrained generator from", pre_trained_ckpt)

    self.discriminator = AnimeDiscriminator()
    self.lsgan_loss = LSGanLoss()
    self.pretrained = VGGPreTrained()
    self.l1_loss = nn.L1Loss('mean')
    self.huber_loss = nn.SmoothL1Loss('mean')

  def inner_training_step(self, batch: Dict[str, torch.Tensor]):
    input_photo = batch['real_data']
    input_cartoon = batch['anime_data']
    anime_gray_data = batch['anime_gray_data']
    anime_smooth_gray_data = batch['anime_smooth_gray_data']

    generated = self.generator(input_photo)

    anime_logit = self.discriminator(input_cartoon)
    anime_gray_logit = self.discriminator(anime_gray_data)
    generated_logit = self.discriminator(generated)
    smooth_logit = self.discriminator(anime_smooth_gray_data)

    # 利用灰度化的图像学习style特征
    c_loss, s_loss = self.con_sty_loss(input_photo, anime_gray_data, generated)
    c_loss = self.hparams.con_weight * c_loss
    s_loss = self.hparams.sty_weight * s_loss
    tv_loss = self.hparams.tv_weight * variation_loss(generated)
    col_loss = self.color_loss(input_photo, generated) * self.hparams.color_weight
    g_loss = (self.hparams.g_adv_weight * self.generator_loss(generated_logit))
    (d_real_loss, d_gray_loss, d_fake_loss, d_real_blur_loss) = self.discriminator_loss(
        anime_logit, anime_gray_logit,
        generated_logit, smooth_logit)
    d_loss_total = (self.hparams.d_adv_weight * (d_real_loss +
                                                 d_fake_loss +
                                                 d_gray_loss +
                                                 d_real_blur_loss))
    g_loss_total = c_loss + s_loss + col_loss + g_loss + tv_loss
    self.log_dict({
        'dis/d_loss': d_loss_total,
        'dis/d_real_loss': d_real_loss,
        'dis/d_fake_loss': d_fake_loss,
        'dis/d_gray_loss': d_gray_loss,
        'dis/d_real_blur_loss': d_real_blur_loss,
        'gen/g_loss': g_loss,
        'gen/c_loss': c_loss,
        'gen/s_loss': s_loss,
        'gen/col_loss': col_loss,
        'gen/tv_loss': tv_loss})
    return d_loss_total, g_loss_total

  def training_step(self, batch: Dict[str, torch.Tensor], batch_idx, optimizer_idx):
    if optimizer_idx == 0:
      d_loss_total, _ = self.inner_training_step(batch)
      return d_loss_total

    elif optimizer_idx == 1:  # train generator
      _, g_loss_total = self.inner_training_step(batch)
      return g_loss_total

  def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx):
    input_photo = batch['real_data']
    input_cartoon = batch['anime_data']
    anime_gray_data = batch['anime_gray_data']
    anime_smooth_gray_data = batch['anime_smooth_gray_data']
    log_images(self, {'input/real': input_photo,
                      'input/anime': input_cartoon,
                      'input/gray': anime_gray_data,
                      'input/smooth_gray': anime_smooth_gray_data,
                      'generate/anime': self.generator(input_photo)})


if __name__ == "__main__":
  run_train(AnimeGANv2, WhiteBoxGanDataModule)
