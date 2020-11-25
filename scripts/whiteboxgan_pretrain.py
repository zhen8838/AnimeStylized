import os
import sys
sys.path.insert(0, os.getcwd())
from networks.gan import SpectNormDiscriminator, UnetGenerator
from networks.pretrainnet import VGGCaffePreTrained
from datamodules.whiteboxgands import WhiteBoxGANDataModule
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from scripts.common import run_common, log_images
from scripts.whiteboxgan import WhiteBoxGAN


class WhiteBoxGANPretrain(WhiteBoxGAN):
  def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
    input_cartoon, input_photo = batch
    generator_img = self.generator(input_photo)
    recon_loss = self.l1_loss(input_photo, generator_img)
    return recon_loss

  def configure_optimizers(self):
    lr_g = self.hparams.lr_g
    b1 = self.hparams.b1
    b2 = self.hparams.b2

    opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(b1, b2))
    return opt_g

  def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
    input_photo = torch.cat(batch)
    generator_img = self.generator(input_photo)

    log_images(self, {
        'input/real': input_photo,
        'generate/anime': generator_img,
    }, 8)


if __name__ == "__main__":
  run_common(WhiteBoxGANPretrain, WhiteBoxGANDataModule)
