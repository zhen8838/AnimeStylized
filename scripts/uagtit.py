import os
import sys
sys.path.insert(0, os.getcwd())
from networks.gan import ResnetGenerator, AttentionDiscriminator, RhoClipper, WClipper
from networks.pretrainnet import FacePreTrained
import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Tuple
from datamodules.uagtitds import UagtitGanDataSet
from scripts.common import run_common, log_images
import itertools
from scripts.whiteboxgan import infer_fn


class UagtitGAN(pl.LightningModule):
  def __init__(self,
               lr: float = 0.0001,
               adv_weight: float = 1,
               cycle_weight: float = 50,
               identity_weight: float = 10,
               cam_weight: float = 1000,
               faceid_weight: float = 1,
               rho_clipper: float = 1,
               w_clipper: float = 1,
               ch: float = 32,
               light: bool = True
               ) -> None:
    super().__init__()
    self.save_hyperparameters()
    self.genA2B = ResnetGenerator(ngf=ch, img_size=256, light=light)
    self.genB2A = ResnetGenerator(ngf=ch, img_size=256, light=light)
    self.disGA = AttentionDiscriminator(input_nc=3, ndf=ch, n_layers=7)
    self.disGB = AttentionDiscriminator(input_nc=3, ndf=ch, n_layers=7)
    self.disLA = AttentionDiscriminator(input_nc=3, ndf=ch, n_layers=5)
    self.disLB = AttentionDiscriminator(input_nc=3, ndf=ch, n_layers=5)
    self.facenet = FacePreTrained('models/model_mobilefacenet.pth')

    self.Rho_clipper = RhoClipper(0, rho_clipper)
    self.W_Clipper = WClipper(0, w_clipper)

    self.L1_loss = nn.L1Loss()
    self.MSE_loss = nn.MSELoss()
    self.BCE_loss = nn.BCEWithLogitsLoss()

  def forward(self, x):
    return self.genA2B(x)[0]

  def on_train_start(self) -> None:
    self.facenet.setup(self.device)

  def configure_optimizers(self):
    G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(
    ), self.genB2A.parameters()), lr=self.hparams.lr, betas=(0.5, 0.999), weight_decay=0.0001)
    D_optim = torch.optim.Adam(
        itertools.chain(self.disGA.parameters(), self.disGB.parameters(),
                        self.disLA.parameters(), self.disLB.parameters()),
        lr=self.hparams.lr, betas=(0.5, 0.999), weight_decay=0.0001
    )
    return D_optim, G_optim

  def training_step(self, batch: Dict[str, torch.Tensor], batch_idx, optimizer_idx):
    real_A, real_B = batch
    if optimizer_idx == 0:  # Update D
      fake_A2B, _, _ = self.genA2B(real_A)
      fake_B2A, _, _ = self.genB2A(real_B)

      real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
      real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
      real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
      real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

      fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
      fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
      fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
      fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

      D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + \
          self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))

      D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + \
          self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))

      D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + \
          self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))

      D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) +\
          self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))

      D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + \
          self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))

      D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + \
          self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))

      D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + \
          self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))

      D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) +\
          self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

      D_loss_A = self.hparams.adv_weight * (D_ad_loss_GA +
                                            D_ad_cam_loss_GA +
                                            D_ad_loss_LA + D_ad_cam_loss_LA)
      D_loss_B = self.hparams.adv_weight * (D_ad_loss_GB +
                                            D_ad_cam_loss_GB +
                                            D_ad_loss_LB + D_ad_cam_loss_LB)
      self.log_dict({
          'dis/A/D_ad_loss_GA': D_ad_loss_GA,
          'dis/A/D_ad_cam_loss_GA': D_ad_cam_loss_GA,
          'dis/A/D_ad_loss_LA': D_ad_loss_LA,
          'dis/A/D_ad_cam_loss_LA': D_ad_cam_loss_LA,
          'dis/B/D_ad_loss_GB': D_ad_loss_GB,
          'dis/B/D_ad_cam_loss_GB': D_ad_cam_loss_GB,
          'dis/B/D_ad_loss_LB': D_ad_loss_LB,
          'dis/B/D_ad_cam_loss_LB': D_ad_cam_loss_LB,
      })
      Discriminator_loss = D_loss_A + D_loss_B
      return Discriminator_loss
    elif optimizer_idx == 1:  # Update G
      fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
      fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

      fake_A2B2A, _, _ = self.genB2A(fake_A2B)
      fake_B2A2B, _, _ = self.genA2B(fake_B2A)

      fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
      fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

      fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
      fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
      fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
      fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

      G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
      G_ad_cam_loss_GA = self.MSE_loss(
          fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
      G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
      G_ad_cam_loss_LA = self.MSE_loss(
          fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
      G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
      G_ad_cam_loss_GB = self.MSE_loss(
          fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
      G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
      G_ad_cam_loss_LB = self.MSE_loss(
          fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

      G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
      G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

      G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
      G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

      G_id_loss_A = self.facenet.cosine_distance(real_A, fake_A2B)
      G_id_loss_B = self.facenet.cosine_distance(real_B, fake_B2A)

      G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + \
          self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
      G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + \
          self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

      G_loss_A = (self.hparams.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) +
                  self.hparams.cycle_weight * G_recon_loss_A + self.hparams.identity_weight * G_identity_loss_A +
                  self.hparams.cam_weight * G_cam_loss_A + self.hparams.faceid_weight * G_id_loss_A)
      G_loss_B = (self.hparams.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) +
                  self.hparams.cycle_weight * G_recon_loss_B + self.hparams.identity_weight * G_identity_loss_B +
                  self.hparams.cam_weight * G_cam_loss_B + self.hparams.faceid_weight * G_id_loss_B
                  )
      Generator_loss = G_loss_A + G_loss_B
      self.log_dict({
          'gen/A/G_ad_loss_GA': G_ad_loss_GA,
          'gen/A/G_ad_cam_loss_GA': G_ad_cam_loss_GA,
          'gen/A/G_ad_loss_LA': G_ad_loss_LA,
          'gen/A/G_ad_cam_loss_LA': G_ad_cam_loss_LA,
          'gen/A/G_recon_loss_A': G_recon_loss_A,
          'gen/A/G_identity_loss_A': G_identity_loss_A,
          'gen/A/G_cam_loss_A': G_cam_loss_A,
          'gen/A/G_id_loss_A': G_id_loss_A,
          'gen/A/G_ad_loss_GA': G_ad_loss_GA,
          'gen/B/G_ad_cam_loss_GB': G_ad_cam_loss_GB,
          'gen/B/G_ad_loss_LB': G_ad_loss_LB,
          'gen/B/G_ad_cam_loss_LB': G_ad_cam_loss_LB,
          'gen/B/G_recon_loss_B': G_recon_loss_B,
          'gen/B/G_identity_loss_B': G_identity_loss_B,
          'gen/B/G_cam_loss_B': G_cam_loss_B,
          'gen/B/G_id_loss_B': G_id_loss_B,
      })
      """ log and apply clip """
      self.genA2B.apply(self.Rho_clipper)
      self.genB2A.apply(self.Rho_clipper)

      self.genA2B.apply(self.W_Clipper)
      self.genB2A.apply(self.W_Clipper)
      return Generator_loss

  def validation_step(self, batch, batch_idx):
    real_A, real_B = batch
    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
    # fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
    # fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
    log_images(self, {'gen/A/fake_A2B': fake_A2B})


if __name__ == "__main__":
  run_common(UagtitGAN, UagtitGanDataSet, infer_fn)
