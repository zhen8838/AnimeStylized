import os
import sys
sys.path.insert(0, os.getcwd())
import pytorch_lightning as pl
from networks import PRETRAINEDS, NETWORKS, build_network
from networks.pretrainnet import featrue_extract_wrapper
from datamodules.feature_reconds import FeatrueReconDataModule
import torch
import torch.nn as nn
from scripts.common import run_common, log_images
from typing import List, Tuple


class FeatureRecon(pl.LightningModule):

  def __init__(self, lr_g: float, layer_keys: List[int],
               pretrained_name: str = 'VGGPreTrained',
               pretrained_kwargs: dict = {},
               generator_name: str = 'AnimeGeneratorLite',
               generator_kwargs: dict = {}
               ):
    super().__init__()
    self.save_hyperparameters()

    # self.catoon_generators = nn.ModuleList([AnimeGeneratorLite() for i in layer_keys])
    self.real_generators = nn.ModuleList(
        [build_network(NETWORKS, generator_name, generator_kwargs) for i in layer_keys])

    self.pretraineds = []
    self.name_list: List[str] = []
    for output_index in layer_keys:
      pretrained = build_network(PRETRAINEDS, pretrained_name, pretrained_kwargs)
      # NOTE this script only for one featrue experiment
      name, _ = featrue_extract_wrapper(pretrained, output_index)
      self.name_list.append(name[0])
      self.pretraineds.append(pretrained)
    self.pretraineds = nn.ModuleList(self.pretraineds)
    self.l1_loss = nn.L1Loss()

  def on_fit_start(self):
    for pretrained in self.pretraineds:
      pretrained.setup(self.device)

  def training_step(self, batch: Tuple[torch.Tensor], batch_idx, optimizer_idx):
    input_photo, input_cartoon = batch

    # generated = self.catoon_generators[optimizer_idx](input_cartoon)
    # real_feature_map = self.vggs[optimizer_idx](input_cartoon)
    # fake_feature_map = self.vggs[optimizer_idx](generated)
    # c_cartoon_loss = self.l1_loss(real_feature_map, fake_feature_map)

    generated = self.real_generators[optimizer_idx](input_photo)
    real_feature_map = self.pretraineds[optimizer_idx](input_photo)
    fake_feature_map = self.pretraineds[optimizer_idx](generated)
    c_real_loss = self.l1_loss(real_feature_map, fake_feature_map)
    # loss = c_cartoon_loss + c_real_loss
    loss = c_real_loss
    # self.log_dict(
    #     {f'gen/con_{self.vgg_names[idx-1]}_cartoon_loss': c_cartoon_loss,
    #      f'gen/con_{self.vgg_names[idx-1]}_real_loss': c_real_loss})
    self.log_dict(
        {f'gen/con_{self.name_list[optimizer_idx]}_real_loss': c_real_loss})
    return loss

  def configure_optimizers(self):
    # opt_g = [torch.optim.Adam(
    #     concat((self.catoon_generators[i].parameters(),
    #             self.real_generators[i].parameters())),
    #     lr=self.hparams.lr_g,
    #     betas=(0.5, 0.999)) for i, _ in enumerate(self.hparams.layer_keys)]
    opt_g = [torch.optim.Adam(self.real_generators[i].parameters(),
                              lr=self.hparams.lr_g,
                              betas=(0.5, 0.999)) for i, _ in enumerate(self.hparams.layer_keys)]
    return opt_g

  def validation_step(self, batch, batch_idx):
    input_photo, input_cartoon = batch
    d = {}
    d['input/real'] = input_photo
    d['input/cartoon'] = input_cartoon
    for i, idx in enumerate(self.hparams.layer_keys):
      d[f'gen/{self.name_list[i]}_real'] = self.real_generators[i](input_photo)
      d[f'gen/{self.name_list[i]}_cartoon'] = self.real_generators[i](input_cartoon)
    log_images(self, d)


if __name__ == "__main__":
  run_common(FeatureRecon, FeatrueReconDataModule)
