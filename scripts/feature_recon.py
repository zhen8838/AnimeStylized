import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Tuple
from networks import get_network, get_pretrain
from datamodules.feature_reconds import FeatrueReconDataModule
from utils.extractor import Extractor
from scripts.common import run_common, log_images


class FeatureRecon(pl.LightningModule):

  def __init__(self, lr_g: float, layer_keys: List[int],
               pretrained: dict = {'name': 'VGGPreTrained'},
               generator: str = {'name': 'AnimeGeneratorLite'},
               ):
    super().__init__()
    self.save_hyperparameters()

    self.real_generators = nn.ModuleList(
        [get_network(generator) for i in layer_keys])

    self.pretraineds = []
    self.name_list: List[str] = []
    for output_index in layer_keys:
      pretrained_net = get_pretrain(pretrained)
      # NOTE this script only for one featrue experiment
      name, _ = Extractor(output_index)(pretrained_net)
      self.name_list.append(name[0])
      self.pretraineds.append(pretrained_net)
    self.pretraineds = nn.ModuleList(self.pretraineds)
    self.l1_loss = nn.L1Loss()

  def on_fit_start(self):
    for pretrained in self.pretraineds:
      pretrained.setup(self.device)

  def training_step(self, batch: Tuple[torch.Tensor], batch_idx, optimizer_idx):
    input_photo, input_cartoon = batch
    generated = self.real_generators[optimizer_idx](input_photo)
    real_feature_map = self.pretraineds[optimizer_idx](input_photo)
    fake_feature_map = self.pretraineds[optimizer_idx](generated)
    c_real_loss = self.l1_loss(real_feature_map, fake_feature_map)
    loss = c_real_loss
    self.log_dict(
        {f'gen/con_{self.name_list[optimizer_idx]}_real_loss': c_real_loss})
    return loss

  def configure_optimizers(self):
    opt_g = [torch.optim.Adam(self.real_generators[i].parameters(),
                              lr=self.hparams.lr_g,
                              betas=(0.5, 0.999)) for i, _ in enumerate(self.hparams.layer_keys)]
    return opt_g

  def validation_step(self, batch, batch_idx):
    input_photo, input_cartoon = batch
    d = {}
    d['input/images'] = torch.cat((input_photo, input_cartoon))
    for i, idx in enumerate(self.hparams.layer_keys):
      d[f'gen/{self.name_list[i]}'] = torch.cat(
          (self.real_generators[i](input_photo),
           self.real_generators[i](input_cartoon)))
    log_images(self, d, num=8)


if __name__ == "__main__":
  run_common(FeatureRecon, FeatrueReconDataModule)
