import os
import sys
sys.path.insert(0, os.getcwd())
import pytorch_lightning as pl
from networks.gan import SpectNormDiscriminator, UnetGenerator
from networks.pretrainnet import VGGCaffePreTrained,featrue_extract_wrapper
from datamodules.whiteboxgands import WhiteBoxGANDataModule
from losses.gan_loss import LSGanLoss
import torch
import torch.nn as nn
import torch.nn.functional as nf
import numpy as np
from joblib import Parallel, delayed
import itertools
from torch.distributions import Distribution
from scripts.common import run_common, log_images
from typing import List, Tuple
from utils.superpix import slic, adaptive_slic, sscolor
from functools import partial


def simple_superpixel(batch_image: np.ndarray, superpixel_fn: callable) -> np.ndarray:
  """ convert batch image to superpixel

  Args:
      batch_image (np.ndarray): np.ndarry, shape must be [b,h,w,c]
      seg_num (int, optional): . Defaults to 200.

  Returns:
      np.ndarray: superpixel array, shape = [b,h,w,c]
  """
  num_job = batch_image.shape[0]
  batch_out = Parallel(n_jobs=num_job)(delayed(superpixel_fn)
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

  def setup(self, device: torch.device):
    # NOTE 原论文输入的bgr图像，此处需要改为rgb
    if self.mode == 'normal':
      self.dist = torch.distributions.Normal(
          torch.tensor((0.299, 0.587, 0.114), device=device),
          torch.tensor((0.1, 0.1, 0.1), device=device))
    elif self.mode == 'uniform':
      self.dist = torch.distributions.Uniform(
          torch.tensor((0.199, 0.487, 0.014), device=device),
          torch.tensor((0.399, 0.687, 0.214), device=device))

  def forward(self, *img: torch.Tensor) -> Tuple[torch.Tensor]:
    rgb = self.dist.sample()
    return ((im * rgb[None, :, None, None]) / rgb.sum() for im in img)


class VariationLoss(nn.Module):
  def __init__(self, k_size: int) -> None:
    super().__init__()
    self.k_size = k_size

  def forward(self, image: torch.Tensor):
    b, c, h, w = image.shape
    tv_h = torch.mean((image[:, :, self.k_size:, :] - image[:, :, : -self.k_size, :])**2)
    tv_w = torch.mean((image[:, :, :, self.k_size:] - image[:, :, :, : -self.k_size])**2)
    tv_loss = (tv_h + tv_w) / (3 * h * w)
    return tv_loss


class WhiteBoxGAN(pl.LightningModule):
  SuperPixelDict = {
      'slic': slic,
      'adaptive_slic': adaptive_slic,
      'sscolor': sscolor}

  def __init__(
      self,
      lr_g: float = 2e-4,
      lr_d: float = 2e-4,
      b1: float = 0.5,
      b2: float = 0.99,
      tv_weight: float = 10000.0,
      g_blur_weight: float = 0.1,
      g_gray_weight: float = 0.1,
      recon_weight: float = 200,
      superpixel_fn: str = 'sscolor',
      superpixel_kwarg: dict = {'seg_num': 200},
      pre_trained_ckpt: str = None,
      **kwargs
  ):
    super().__init__()
    self.save_hyperparameters()

    # networks
    self.generator = UnetGenerator()
    self.pre_trained_ckpt = pre_trained_ckpt
    self.disc_gray = SpectNormDiscriminator()
    self.disc_blur = SpectNormDiscriminator()
    self.guided_filter = GuidedFilter()
    self.lsgan_loss = LSGanLoss()
    self.colorshift = ColorShift()
    self.pretrained = VGGCaffePreTrained()
    featrue_extract_wrapper(self.pretrained, 26) # stop the self.pretrained forward progress
    self.l1_loss = nn.L1Loss('mean')
    self.variation_loss = VariationLoss(1)
    self.superpixel_fn = partial(self.SuperPixelDict[superpixel_fn],
                                 **superpixel_kwarg)

  def setup(self, stage: str):
    if stage == 'fit':
      if self.pre_trained_ckpt:
        ckpt = torch.load(self.pre_trained_ckpt)
        generatordict = dict(filter(lambda k: 'generator' in k[0], ckpt['state_dict'].items()))
        generatordict = {k.split('.', 1)[1]: v for k, v in generatordict.items()}
        self.generator.load_state_dict(generatordict, True)
        del ckpt
        del generatordict
        print("Success load pretrained generator from", self.pre_trained_ckpt)
    elif stage == 'test':
      pass

  def on_fit_start(self):
    self.colorshift.setup(self.device)
    self.pretrained.setup(self.device)

  def forward(self, input_photo) -> torch.Tensor:
    generator_img = self.generator(input_photo)
    output = self.guided_filter(input_photo, generator_img, r=1)
    return output

  def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx, optimizer_idx):
    input_cartoon, input_photo = batch

    if optimizer_idx == 0:  # train generator
      generator_img = self.generator(input_photo)
      output = self.guided_filter(input_photo, generator_img, r=1)
      # 1. blur for Surface Representation
      blur_fake = self.guided_filter(output, output, r=5, eps=2e-1)
      blur_fake_logit = self.disc_blur(blur_fake)
      g_loss_blur = self.hparams.g_blur_weight * self.lsgan_loss._g_loss(blur_fake_logit)

      # 2. gray for Textural Representation
      gray_fake, = self.colorshift(output)
      gray_fake_logit = self.disc_gray(gray_fake)
      g_loss_gray = self.hparams.g_gray_weight * self.lsgan_loss._g_loss(gray_fake_logit)

      # 3. superpixel for Structure Representation
      input_superpixel = torch.from_numpy(
          simple_superpixel(output.detach().permute((0, 2, 3, 1)).cpu().numpy(),
                            self.superpixel_fn)
      ).to(self.device).permute((0, 3, 1, 2))

      vgg_output = self.pretrained(output)
      _, c, h, w = vgg_output.shape
      vgg_superpixel = self.pretrained(input_superpixel)
      superpixel_loss = (self.hparams.recon_weight *
                         self.l1_loss(vgg_superpixel, vgg_output) / (c * h * w))

      # 4. Content loss
      vgg_photo = self.pretrained(input_photo)
      photo_loss = self.hparams.recon_weight * self.l1_loss(vgg_photo, vgg_output) / (c * h * w)

      # 5. total variation loss
      tv_loss = self.hparams.tv_weight * self.variation_loss(output)

      g_loss_total = tv_loss + g_loss_blur + g_loss_gray + superpixel_loss + photo_loss
      self.log_dict({'gen/g_loss': g_loss_total,
                     'gen/tv_loss': tv_loss,
                     'gen/g_blur': g_loss_blur,
                     'gen/g_gray': g_loss_gray,
                     'gen/photo_loss': photo_loss,
                     'gen/superpixel_loss': superpixel_loss,
                     })

      return g_loss_total
    elif optimizer_idx == 1:  # train discriminator
      generator_img = self.generator(input_photo)
      output = self.guided_filter(input_photo, generator_img, r=1)
      # 1. blur for Surface Representation
      blur_fake = self.guided_filter(output, output, r=5, eps=2e-1)
      blur_cartoon = self.guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)
      blur_real_logit = self.disc_blur(blur_cartoon)
      blur_fake_logit = self.disc_blur(blur_fake)
      d_loss_blur = self.lsgan_loss._d_loss(blur_real_logit, blur_fake_logit)

      # 2. gray for Textural Representation
      gray_fake, gray_cartoon = self.colorshift(output, input_cartoon)
      gray_real_logit = self.disc_gray(gray_cartoon)
      gray_fake_logit = self.disc_gray(gray_fake)
      d_loss_gray = self.lsgan_loss._d_loss(gray_real_logit, gray_fake_logit)

      d_loss_total = d_loss_blur + d_loss_gray
      self.log_dict({'dis/d_loss': d_loss_total,
                     'dis/d_blur': d_loss_blur,
                     'dis/d_gray': d_loss_gray})

      return d_loss_total

  def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
    input_photo = torch.cat(batch)
    generator_img = self.generator(input_photo)
    output = self.guided_filter(input_photo, generator_img, r=1)
    blur_fake = self.guided_filter(output, output, r=5, eps=2e-1)
    gray_fake, = self.colorshift(output)
    input_superpixel = torch.from_numpy(
        simple_superpixel(output.detach().permute((0, 2, 3, 1)).cpu().numpy(),
                          self.superpixel_fn)
    ).to(self.device).permute((0, 3, 1, 2))

    log_images(self, {
        'input/real': input_photo,
        'input/superpix': input_superpixel,
        'generate/anime': generator_img,
        'generate/filtered': output,
        'generate/gray': gray_fake,
        'generate/blur': blur_fake,
    }, num=8)

  def configure_optimizers(self):
    lr_g = self.hparams.lr_g
    lr_d = self.hparams.lr_d
    b1 = self.hparams.b1
    b2 = self.hparams.b2

    opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(b1, b2))
    opt_d = torch.optim.Adam(itertools.chain(self.disc_blur.parameters(),
                                             self.disc_gray.parameters()),
                             lr=lr_d, betas=(b1, b2))
    return [opt_g, opt_d], []


def infer_fn(model: WhiteBoxGAN, image_path: str):
  from datamodules.dsfunction import imread, denormalize
  import datamodules.dstransform as transforms
  from pathlib import Path
  import cv2
  from utils.terminfo import INFO

  infer_transform = transforms.Compose([
      transforms.ResizeToScale((256, 256), 32),
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
  ])
  model.setup('test')
  model.eval()

  def infer_one_image(image_path: Path, output_root: partial):
    im = imread(image_path.as_posix())
    feed_im = infer_transform(im)
    out_im = model.forward(feed_im[None, ...])[0]
    draw_im = (denormalize(out_im.permute((1, 2, 0)).detach().numpy()) * 255).astype('uint8')
    output_path = output_root / (image_path.stem + '_out' + image_path.suffix)
    cv2.imwrite(output_path.as_posix(), cv2.cvtColor(draw_im, cv2.COLOR_RGB2BGR))
    print(INFO, 'Convert', image_path, 'to', output_path)

  image_path = Path(image_path)
  if image_path.is_file():
    output_root = image_path.parent
    infer_one_image(image_path, output_root)
  elif image_path.is_dir():
    output_root: Path = image_path.parent / (image_path.name + '_out')
    if not output_root.exists():
      output_root.mkdir()
    for p in image_path.iterdir():
      infer_one_image(p, output_root)


if __name__ == "__main__":
  run_common(WhiteBoxGAN, WhiteBoxGANDataModule, infer_fn)
