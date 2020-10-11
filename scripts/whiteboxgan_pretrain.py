import os
import sys
sys.path.insert(0, os.getcwd())
import pytorch_lightning as pl
from networks.whiteboxgan import SpectNormDiscriminator, UnetGenerator, VGGPreTrained
from datasets.whiteboxgan import WhiteBoxGanDataModule, denormalize
from losses.gan_loss import LSGanLoss
from typing import Dict, List
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from scripts.common import run_train


class GAN(pl.LightningModule):

  def __init__(
      self,
      lr: float = 2e-4,
      b1: float = 0.5,
      b2: float = 0.99,
      **kwargs
  ):
    super().__init__()
    self.save_hyperparameters()

    # networks
    self.generator = UnetGenerator()
    self.l1_loss = nn.L1Loss('mean')
    self.pretrained = VGGPreTrained()
    self.pretrained.freeze()

  def forward(self, im):
    return self.generator(im)

  def on_train_start(self) -> None:
    self.pretrained.setup(self.device)

  def training_step(self, batch: Dict[str, torch.Tensor], batch_idx):
    input_photo = batch['real_data']
    input_cartoon = batch['anime_data']
    # anime_gray_data = batch['anime_gray_data']
    # anime_smooth_data = batch['anime_smooth_data']

    generator_img = self.generator(input_photo)

    vgg_photo = self.pretrained(input_photo)
    vgg_output = self.pretrained(generator_img)

    vgg_recon_loss = self.l1_loss(vgg_photo, vgg_output)
    recon_loss = self.l1_loss(generator_img, input_photo)
    total_loss = vgg_recon_loss + recon_loss

    self.log_dict({'loss': total_loss,
                   'vgg_loss': vgg_recon_loss,
                   'recon_loss': recon_loss})

    if batch_idx % 50 == 0:
      input_photo_show = torchvision.utils.make_grid(input_photo[:4], nrow=4)
      generator_img_show = torchvision.utils.make_grid(generator_img[:4], nrow=4)
      input_photo_show = denormalize(input_photo_show)
      generator_img_show = denormalize(generator_img_show)

      tb: SummaryWriter = self.logger.experiment
      tb.add_image('input_photo', input_photo_show, batch_idx)
      tb.add_image('generator_img', generator_img_show, batch_idx)

    return total_loss

  def configure_optimizers(self):
    lr = self.hparams.lr
    b1 = self.hparams.b1
    b2 = self.hparams.b2

    opt = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))

    return [opt], []


if __name__ == "__main__":
  run_train(GAN, WhiteBoxGanDataModule)
