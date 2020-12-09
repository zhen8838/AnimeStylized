import os
import sys
sys.path.insert(0, os.getcwd())
from datamodules.animegands import AnimeGANDataModule
from networks import NETWORKS, PRETRAINEDS
from utils.extractor import Extractor
from losses.lsfunction import variation_loss
from losses.gan_loss import LSGanLoss
from typing import Dict, Tuple
from torch import nn
from scripts.common import run_common
from scripts.animegan import AnimeGAN, infer_fn


class AnimeGANv2Plus(AnimeGAN):

  def __init__(self, lr_g: float = 2e-4,
               lr_d: float = 2e-4,
               g_adv_weight: float = 300.,
               d_adv_weight: float = 300.,
               con_weight: float = 1.5,
               low_con_weight: float = 1,
               high_con_weight: float = 1,
               sty_weight: float = 2.8,
               color_weight: float = 10.,
               pre_trained_ckpt: str = None,
               tv_weight: float = 1.,
               output_keys: Dict[str, str] = {
                   'content': 'features.26',
                   'style': 'features.30'
               },
               generator_name: str = 'AnimeGeneratorLite',
               discriminator_name: str = 'AnimeDiscriminator',
               pretrained_name: str = 'VGGCaffePreTrained',
               **kwargs):
    super().__init__()
    self.save_hyperparameters()

    # networks

    self.generator = NETWORKS.get(generator_name)()
    self.pre_trained_ckpt = pre_trained_ckpt
    self.discriminator = NETWORKS.get(discriminator_name)()
    self.lsgan_loss = LSGanLoss()
    self.pretrained = PRETRAINEDS.get(pretrained_name)()
    self.premap = output_keys
    # stop the self.pretrained forward progress
    Extractor([v for (k, v) in output_keys.items()])(self.pretrained)

    self.l1_loss = nn.L1Loss()
    self.huber_loss = nn.SmoothL1Loss()

  def get_featrue_con_sty(self, image):
    feature_dict = self.pretrained(image)
    content = feature_dict[self.premap['content']]
    style = feature_dict[self.premap['style']]
    return content, style

  def con_sty_loss(self, real, anime, fake):
    real_con, real_sty = self.get_featrue_con_sty(real)
    anime_con, anime_sty = self.get_featrue_con_sty(anime)
    fake_con, fake_sty = self.get_featrue_con_sty(fake)
    c_loss = (self.hparams.low_con_weight * self.l1_loss(real_con, fake_con) +
              self.hparams.high_con_weight * self.l1_loss(real_sty, fake_sty))

    s_loss = self.style_loss(anime_sty, fake_sty)
    return c_loss, s_loss

  def training_step(self, batch, batch_idx, optimizer_idx):
    input_photo, (input_cartoon, anime_gray_data), anime_smooth_gray_data = batch

    if optimizer_idx == 0:  # train discriminator
      generated = self.generator(input_photo)
      anime_logit = self.discriminator(input_cartoon)
      anime_gray_logit = self.discriminator(anime_gray_data)
      generated_logit = self.discriminator(generated)
      smooth_logit = self.discriminator(anime_smooth_gray_data)

      (d_real_loss, d_gray_loss, d_fake_loss, d_real_blur_loss) = self.discriminator_loss(
          anime_logit, anime_gray_logit,
          generated_logit, smooth_logit)
      d_real_loss = self.hparams.d_adv_weight * d_real_loss
      d_gray_loss = self.hparams.d_adv_weight * d_gray_loss
      d_fake_loss = self.hparams.d_adv_weight * d_fake_loss
      d_real_blur_loss = self.hparams.d_adv_weight * d_real_blur_loss
      d_loss_total = d_real_loss + d_fake_loss + d_gray_loss + d_real_blur_loss

      self.log_dict({
          'dis/d_loss': d_loss_total,
          'dis/d_real_loss': d_real_loss,
          'dis/d_fake_loss': d_fake_loss,
          'dis/d_gray_loss': d_gray_loss,
          'dis/d_real_blur_loss': d_real_blur_loss})
      return d_loss_total

    elif optimizer_idx == 1:  # train generator
      generated = self.generator(input_photo)
      generated_logit = self.discriminator(generated)
      c_loss, s_loss = self.con_sty_loss(input_photo,
                                         anime_gray_data,
                                         generated)

      c_loss = self.hparams.con_weight * c_loss
      s_loss = self.hparams.sty_weight * s_loss
      tv_loss = self.hparams.tv_weight * variation_loss(generated)
      col_loss = self.color_loss(input_photo, generated) * self.hparams.color_weight
      g_loss = (self.hparams.g_adv_weight * self.generator_loss(generated_logit))
      g_loss_total = c_loss + s_loss + col_loss + g_loss + tv_loss
      self.log_dict({
          'gen/g_loss': g_loss,
          'gen/c_loss': c_loss,
          'gen/s_loss': s_loss,
          'gen/col_loss': col_loss,
          'gen/tv_loss': tv_loss})

      return g_loss_total


if __name__ == "__main__":
  run_common(AnimeGANv2Plus, AnimeGANDataModule, infer_fn)
