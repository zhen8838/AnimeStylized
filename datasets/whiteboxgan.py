from typing import List, Dict, Tuple
import pytorch_lightning as pl
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.samplers import RandomClipSampler
import random
import cv2
import numpy as np
import torchvision.transforms.functional as tf
import torchvision.transforms as T
from .dsfunction import normalize, denormalize


def imread(path: str):
  return cv2.cvtColor(cv2.imread(path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def reduce_hw(img_hw: List[int], min_hw: List[int]) -> Tuple[int]:
  im_h, im_w = img_hw
  if im_h <= min_hw[0]:
    im_h = min_hw[0]
  else:
    x = im_h % 32
    im_h = im_h - x

  if im_w < min_hw[1]:
    im_w = min_hw[1]
  else:
    y = im_w % 32
    im_w = im_w - y
  return (im_h, im_w)


class AnimeGanDataSet(Dataset):
  def __init__(self, root: str, style: str,
               train=True, augment=True, normalize=True, totenor=True):
    super().__init__()
    self.root = Path(root)
    self.augment = augment
    self.normalize = normalize
    self.totenor = totenor
    assert self.root.exists(), 'dataset_root not exists !'
    self.train = train
    self.real_imgs: List[str] = None
    self.anime_imgs: List[str] = None
    self.sooth_imgs: List[str] = None
    self.val_imgs: List[str] = None
    self.anime_img_nums: int = None
    if self.train:
      trian_root = (self.root / 'train_photo')
      self.real_imgs = [(trian_root / p).as_posix() for p in trian_root.iterdir()]

      anime_root = (self.root / f'{style}/style')
      self.anime_imgs: List[str] = [(anime_root / p).as_posix() for p in anime_root.iterdir()]
      self.sooth_imgs: List[str] = [s.replace('style', 'smooth', 1) for s in self.anime_imgs]
      self.anime_img_nums = len(self.anime_imgs)
    else:
      val_root = (self.root / 'test/test_photo')
      self.val_imgs = [(val_root / p).as_posix() for p in val_root.iterdir()]

  def __len__(self):
    if self.train:
      return len(self.real_imgs)
    else:
      return len(self.val_imgs)

  def __getitem__(self, index) -> Dict[str, torch.Tensor]:
    if self.train:
      real_path = self.real_imgs[index]
      randidx = random.randint(0, self.anime_img_nums - 1)
      anime_path = self.anime_imgs[randidx]
      randidx = random.randint(0, self.anime_img_nums - 1)
      sooth_path = self.sooth_imgs[randidx]
      d: dict = self.process_train(real_path, anime_path, sooth_path)
    else:
      real_path = self.val_imgs[index]
      d = self.process_val(real_path)
    return d

  def do_normalize(self, d: dict):
    if self.normalize:
      for k in d.keys():
        d[k] = normalize(d[k])
    return d

  def do_augment(self, d: dict):
    if self.augment:
      k = 'real_data'
      if random.random() < 0.5:
        d[k] = cv2.flip(d[k], 1)
    return d

  def do_totensor(self, d: dict):
    """ to tensor will convert image to 0~1 """
    if self.totenor:
      for k, v in d.items():
        d[k] = tf.to_tensor(v)
    return d

  def do_grayscale(self, image: np.ndarray) -> np.ndarray:
    """ rgb to grayscale keep 3 channel """
    return np.tile(np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), -1), [1, 1, 3])

  def process_train(self, real_path, anime_path, anime_smooth_path) -> Dict[str, torch.Tensor]:
    real = imread(real_path)
    anime = imread(anime_path)
    anime_smooth = imread(anime_smooth_path)
    d = dict(real_data=real,
             anime_data=anime,
             anime_gray_data=self.do_grayscale(anime),
             anime_smooth_gray_data=self.do_grayscale(anime_smooth))
    d = self.do_normalize(self.do_totensor(self.do_augment(d)))
    return d

  def process_val(self, real_path) -> Dict[str, torch.Tensor]:
    real: np.ndarray = imread(real_path)
    hw = reduce_hw(real.shape[:2], [256, 256])
    d = dict(real_data=cv2.resize(real, hw))
    d = self.do_normalize(self.do_totensor(self.do_augment(d)))
    return d


class WhiteBoxGanDataModule(pl.LightningDataModule):
  def __init__(self, root: str, style: str, batch_size: int = 8, num_workers: int = 4,
               augment=True, normalize=True, totenor=True):
    super().__init__()
    self.root = root
    self.style = style
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.augment = augment
    self.normalize = normalize
    self.totenor = totenor
    self.dims = (3, 256, 256)

  def setup(self, stage=None):
    if stage == 'fit':
      self.ds_train = AnimeGanDataSet(self.root, self.style, train=True,
                                      augment=self.augment,
                                      normalize=self.normalize,
                                      totenor=self.totenor)
      self.ds_val = AnimeGanDataSet(self.root, self.style,
                                    train=False,
                                    augment=self.augment,
                                    normalize=self.normalize,
                                    totenor=self.totenor)
    else:
      self.ds_test = AnimeGanDataSet(self.root, self.style, train=False,
                                     augment=self.augment,
                                     normalize=self.normalize,
                                     totenor=self.totenor)

  def train_dataloader(self):
    return DataLoader(
        self.ds_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

  def val_dataloader(self):
    return DataLoader(self.ds_val, shuffle=True,
                      batch_size=4, num_workers=4)

  def test_dataloader(self):
    return DataLoader(self.ds_test, batch_size=4)
