from typing import List, Dict, Tuple
import pytorch_lightning as pl
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from .uagtitds import ImageFolder, MergeDataset, MultiRandomSampler
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
  def __init__(self, root: str, style: str, data_mean: list = [13.1360, -8.6698, -4.4661],
               train=True, augment=True, normalize=True, totenor=True):
    super().__init__()
    self.data_mean = np.array(data_mean, dtype='float32')
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
    anime_smooth = cv2.imread(anime_smooth_path, cv2.IMREAD_GRAYSCALE)
    anime_gray = cv2.imread(anime_path, cv2.IMREAD_GRAYSCALE)
    d = dict(real_data=real,
             anime_data=np.clip(anime + self.data_mean, 0, 255).astype('uint8'),
             anime_gray_data=np.tile(anime_gray[..., None], [1, 1, 3]),
             anime_smooth_gray_data=np.tile(anime_smooth[..., None], [1, 1, 3]))
    d = self.do_normalize(self.do_totensor(self.do_augment(d)))
    return d

  def process_val(self, real_path) -> Dict[str, torch.Tensor]:
    real: np.ndarray = imread(real_path)
    hw = reduce_hw(real.shape[:2], [256, 256])
    d = dict(real_data=cv2.resize(real, hw))
    d = self.do_normalize(self.do_totensor(self.do_augment(d)))
    return d


class WhiteBoxGanDataModule(pl.LightningDataModule):
  def __init__(self, root: str, style: str,
               batch_size: int = 8, num_workers: int = 4,
               augment=True, normalize=True, totenor=True):
    super().__init__()
    self.root = Path(root)
    self.style = style
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.augment = augment
    self.normalize = normalize
    self.totenor = totenor
    self.dims = (3, 256, 256)

    idenity = transforms.Lambda(lambda x: x)
    self.train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if augment else idenity,
        transforms.ToTensor() if totenor else idenity,
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) if normalize else idenity])

    self.train_gray_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomHorizontalFlip() if augment else idenity,
        transforms.ToTensor() if totenor else idenity,
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) if normalize else idenity])

    self.val_transform = transforms.Compose([
        transforms.ToTensor() if totenor else idenity,
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) if normalize else idenity)])

  def setup(self, stage=None):
    if stage == 'fit':
      trian_root = (self.root / 'train_photo')
      anime_root = (self.root / f'{self.style}/style')
      smooth_root = (self.root / f'{self.style}/smooth')
      train_real = ImageFolder(trian_root.as_posix(),
                               transform=self.train_transform)
      train_anime = ImageFolder(anime_root.as_posix(),
                                transform=self.train_transform)
      train_anime_gray = ImageFolder(anime_root.as_posix(),
                                     transform=self.train_gray_transform)
      train_anime = TensorDataset(train_anime, train_anime_gray)
      train_smooth_gray = ImageFolder(smooth_root.as_posix(),
                                      transform=self.train_gray_transform)
      self.ds_train = MergeDataset(train_real, train_anime, train_smooth_gray)

      val_root = (self.root / 'test/test_photo')
      self.ds_val = ImageFolder(val_root.as_posix(),
                                transform=self.val_transform)
    else:
      val_root = (self.root / 'test/test_photo')
      self.ds_val = ImageFolder(val_root.as_posix(),
                                transform=self.val_transform)

  def train_dataloader(self):
    return DataLoader(
        self.ds_train,
        sampler=MultiRandomSampler(self.ds_train),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=True)

  def val_dataloader(self):
    return DataLoader(self.ds_val, shuffle=True,
                      batch_size=4, num_workers=4)

  def test_dataloader(self):
    return DataLoader(self.ds_val, shuffle=True,
                      batch_size=4, num_workers=4)
