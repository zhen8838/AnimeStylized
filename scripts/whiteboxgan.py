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
import torch.nn.functional as nf
import torch.functional as F
import torchvision.transforms.functional as tf
import torchvision
import numpy as np
from skimage import segmentation, color
from joblib import Parallel, delayed
from optimizers import DummyOptimizer
import itertools
from torch.distributions import Distribution
from torch.utils.tensorboard import SummaryWriter
from scripts.common import run_train


def simple_superpixel(batch_image: np.ndarray, seg_num=200) -> np.ndarray:
  """ convert batch image to superpixel

  Args:
      batch_image (np.ndarray): np.ndarry, shape must be [b,h,w,c] 
      seg_num (int, optional): . Defaults to 200.

  Returns:
      np.ndarray: superpixel array, shape = [b,h,w,c]
  """
  def process_slic(image):
    seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1,
                                  compactness=10, convert2lab=True,
                                  start_label=0)
    image = color.label2rgb(seg_label, image, kind='avg', bg_label=-1)
    return image

  num_job = batch_image.shape[0]
  batch_out = Parallel(n_jobs=num_job)(delayed(process_slic)
                                       (image) for image in batch_image)
  return np.array(batch_out)


class GuidedFilter(nn.Module):
  def box_filter(self, x: torch.Tensor, r):
    ch = x.shape[1]
    k = 2 * r + 1
    weight = 1 / ((k)**2)  # 1/9
    # [c,1,3,3] * 1/9
    box_kernel = torch.ones((ch, 1, k, k), dtype=torch.float32, device=x.device).fill_(weight)
    # same padding
    return nf.conv2d(x, box_kernel, padding=r, groups=ch)

  def forward(self, x: torch.Tensor, y: torch.Tensor, r, eps=1e-2):
    b, c, h, w = x.shape
    device = x.device
    # 全1的图像进行滤波的结果
    N = self.box_filter(torch.ones((1, 1, h, w), dtype=x.dtype, device=device), r)

    mean_x = self.box_filter(x, r) / N
    mean_y = self.box_filter(y, r) / N
    cov_xy = self.box_filter(x * y, r) / N - mean_x * mean_y
    var_x = self.box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = self.box_filter(A, r) / N
    mean_b = self.box_filter(b, r) / N

    output = mean_A * x + mean_b
    return output


class ColorShift(nn.Module):
  def __init__(self, mode='uniform'):
    super().__init__()
    self.dist: Distribution = None
    self.mode = mode

    # self.dist:torch.distributions. = dist

  def setup(self, device: torch.device):
    if self.mode == 'normal':
      self.dist = torch.distributions.Normal(
          torch.tensor((0.299, 0.587, 0.114), device=device),
          torch.tensor((0.1, 0.1, 0.1), device=device))
    elif self.mode == 'uniform':
      self.dist = torch.distributions.Uniform(
          torch.tensor((0.014, 0.487, 0.199), device=device),
          torch.tensor((0.214, 0.687, 0.399), device=device))

  def forward(self, img: torch.Tensor):
    rgb = self.dist.sample()
    return img * rgb[None, :, None, None]


class GAN(pl.LightningModule):

  def __init__(
      self,
      lr_g: float = 2e-4,
      lr_d: float = 2e-4,
      b1: float = 0.5,
      b2: float = 0.99,
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

    self.disc_gray = SpectNormDiscriminator()
    self.disc_blur = SpectNormDiscriminator()
    self.guided_filter = GuidedFilter()
    self.lsgan_loss = LSGanLoss()
    self.colorshift = ColorShift()
    self.pretrained = VGGPreTrained()
    self.pretrained.freeze()
    self.l1_loss = nn.L1Loss('mean')

  def on_train_start(self) -> None:
    self.colorshift.setup(self.device)
    self.pretrained.setup(self.device)

  def forward(self, im):
    return self.generator(im)

  def do_disc(self, model, real, fake):
    real_logit = model(real)
    fake_logit = model(fake)
    return real_logit, fake_logit

  @staticmethod
  def variation_loss(image, k_size=1):
    h, w = image.shape[1:3]
    tv_h = torch.mean((image[:, k_size:, :, :] - image[:, :h - k_size, :, :])**2)  # 上下两列交叉相减
    tv_w = torch.mean((image[:, :, k_size:, :] - image[:, :, :w - k_size, :])**2)  # 左右两行交叉相减
    tv_loss = (tv_h + tv_w) / (3 * h * w)
    return tv_loss

  def training_step(self, batch: Dict[str, torch.Tensor], batch_idx, optimizer_idx):
    input_photo = batch['real_data']
    input_cartoon = batch['anime_data']
    # anime_gray_data = batch['anime_gray_data']
    # anime_smooth_data = batch['anime_smooth_data']

    if optimizer_idx == 0:  # get superpixel image
      generator_img = self.generator(input_photo)
      output: torch.Tensor = self.guided_filter(input_photo, generator_img, r=1)

      self.step_input_superpixel: torch.Tensor = torch.from_numpy(
          simple_superpixel(output.detach().cpu().transpose(1, 3).numpy(), seg_num=200)
      ).to(self.device).transpose(1, 3)

    elif optimizer_idx == 1:  # train generator
      generator_img = self.generator(input_photo)
      output = self.guided_filter(input_photo, generator_img, r=1)
      blur_fake = self.guided_filter(output, output, r=5, eps=2e-1)
      blur_cartoon = self.guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)

      gray_fake = self.colorshift(output)
      gray_cartoon = self.colorshift(input_cartoon)
      gray_real_logit, gray_fake_logit = self.do_disc(self.disc_gray, gray_cartoon, gray_fake)
      blur_real_logit, blur_fake_logit = self.do_disc(self.disc_blur, blur_cartoon, blur_fake)
      g_loss_gray = self.lsgan_loss._forward_g_loss(gray_real_logit, gray_fake_logit)
      g_loss_blur = self.lsgan_loss._forward_g_loss(blur_real_logit, blur_fake_logit)

      # d_loss_gray, g_loss_gray = self.gan_disc_loss(self.disc_gray, gray_cartoon, gray_fake)
      # d_loss_blur, g_loss_blur = self.gan_disc_loss(self.disc_blur, blur_cartoon, blur_fake)

      vgg_photo = self.pretrained(input_photo)
      vgg_output = self.pretrained(output)
      vgg_superpixel = self.pretrained(self.step_input_superpixel)

      photo_loss = self.l1_loss(vgg_photo, vgg_output)
      superpixel_loss = self.l1_loss(vgg_superpixel, vgg_output)
      recon_loss = photo_loss + superpixel_loss
      tv_loss = self.variation_loss(output)
      g_loss_total = 1e4 * tv_loss + 1e-1 * g_loss_blur + g_loss_gray + 2e2 * recon_loss
      self.log_dict({'g_loss': g_loss_total,
                     'tv_loss': 1e4 * tv_loss,
                     'g_loss_blur': 1e-1 * g_loss_blur,
                     'g_loss_gray': g_loss_gray,
                     'recon_loss': 2e2 * recon_loss})
      return g_loss_total
    elif optimizer_idx == 2:  # train discriminator
      generator_img = self.generator(input_photo)
      output = self.guided_filter(input_photo, generator_img, r=1)
      blur_fake = self.guided_filter(output, output, r=5, eps=2e-1)
      blur_cartoon = self.guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)

      gray_fake = self.colorshift(output)
      gray_cartoon = self.colorshift(input_cartoon)
      gray_real_logit, gray_fake_logit = self.do_disc(self.disc_gray, gray_cartoon, gray_fake)
      blur_real_logit, blur_fake_logit = self.do_disc(self.disc_blur, blur_cartoon, blur_fake)
      d_loss_gray = self.lsgan_loss._forward_d_loss(gray_real_logit, gray_fake_logit)
      d_loss_blur = self.lsgan_loss._forward_d_loss(blur_real_logit, blur_fake_logit)
      d_loss_total = d_loss_blur + d_loss_gray

      self.log_dict({'d_loss': d_loss_total,
                     'd_loss_blur': d_loss_blur,
                     'd_loss_gray': d_loss_gray})

      if batch_idx % 50 == 0:
        input_photo_show = torchvision.utils.make_grid(input_photo[:4], nrow=4)
        generator_img_show = torchvision.utils.make_grid(generator_img[:4], nrow=4)
        output_show = torchvision.utils.make_grid(output[:4], nrow=4)
        input_cartoon_show = torchvision.utils.make_grid(input_cartoon[:4], nrow=4)
        input_photo_show = denormalize(input_photo_show)
        generator_img_show = denormalize(generator_img_show)
        output_show = denormalize(output_show)
        input_cartoon_show = denormalize(input_cartoon_show)

        tb: SummaryWriter = self.logger.experiment
        tb.add_image('input_photo', input_photo_show, batch_idx)
        tb.add_image('generator_img', generator_img_show, batch_idx)
        tb.add_image('output', output_show, batch_idx)
        tb.add_image('input_cartoon', input_cartoon_show, batch_idx)

      return d_loss_total

  def configure_optimizers(self):
    lr_g = self.hparams.lr_g
    lr_d = self.hparams.lr_d
    b1 = self.hparams.b1
    b2 = self.hparams.b2

    opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(b1, b2))
    opt_d = torch.optim.Adam(itertools.chain(self.disc_blur.parameters(),
                                             self.disc_gray.parameters()), lr=lr_d, betas=(b1, b2))
    return [DummyOptimizer(), opt_g, opt_d], []


if __name__ == "__main__":
  run_train(GAN, WhiteBoxGanDataModule)
