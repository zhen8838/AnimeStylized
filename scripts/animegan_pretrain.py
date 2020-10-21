import os
import sys
sys.path.insert(0, os.getcwd())
import pytorch_lightning as pl
from datasets.whiteboxgan import WhiteBoxGanDataModule, denormalize
from losses.gan_loss import LSGanLoss
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as nf
import torch.functional as F
import torchvision.transforms.functional as tf
from scripts.common import run_train, log_images
from scripts.animegan import AnimeGAN


class AnimeGANPreTrain(AnimeGAN):

  def training_step(self, batch: Tuple[torch.Tensor], batch_idx):
    input_photo = batch[0]

    generated = self.generator(input_photo)

    real_feature_map = self.pretrained(input_photo)
    fake_feature_map = self.pretrained(generated)
    init_c_loss = self.l1_loss(real_feature_map, fake_feature_map)
    loss = self.hparams.con_weight * init_c_loss

    self.log_dict({'loss': loss})

    return loss

  def configure_optimizers(self):
    opt_g = torch.optim.Adam(self.generator.parameters(),
                             lr=self.hparams.lr_g, betas=(0.5, 0.999))
    return [opt_g], []


if __name__ == "__main__":
  run_train(AnimeGANPreTrain, WhiteBoxGanDataModule)
