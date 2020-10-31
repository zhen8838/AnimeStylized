import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datamodules.dataset import (ImageFolder, MergeDataset,
                                 MultiRandomSampler, MultiBatchDataset,
                                 MultiBatchSampler, random_split,
                                 ConcatDataset)
import datamodules.dstransform as transforms
from pathlib import Path


class WhiteBoxGANDataModule(pl.LightningDataModule):
  def __init__(self, root: str,
               scene_style: str = 'shinkai',
               face_style: str = 'pa_face',
               sample_steps: list = [4, 1],
               batch_size: int = 8, num_workers: int = 4,
               normalize=True, totenor=True):
    super().__init__()
    self.root = Path(root)
    self.scene_style = scene_style
    self.face_style = face_style
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.normalize = normalize
    self.totenor = totenor
    self.sample_steps = sample_steps
    self.dims = (3, 256, 256)

    idenity = transforms.Lambda(lambda x: x)

    self.transform = transforms.Compose([
        transforms.ToTensor() if totenor else idenity,
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) if normalize else idenity])

  def setup(self, stage=None):
    scenery_cartoon = ImageFolder((self.root / f'scenery_cartoon/{self.scene_style}').as_posix(),
                                  transform=self.transform)
    scenery_photo = ImageFolder((self.root / 'scenery_photo').as_posix(),
                                transform=self.transform)
    n_scenery = len(scenery_photo)
    scenery_photo_train, scenery_photo_val = random_split(scenery_photo,
                                                          [int(n_scenery * 0.9),
                                                           n_scenery - int(n_scenery * 0.9)])

    scenery_ds = MergeDataset(scenery_cartoon, scenery_photo_train)

    face_cartoon = ImageFolder((self.root / f'face_cartoon/{self.face_style}').as_posix(),
                               transform=self.transform)
    face_photo = ImageFolder((self.root / 'face_photo').as_posix(),
                             transform=self.transform)
    n_face = len(face_photo)
    face_photo_train, face_photo_val = random_split(face_photo,
                                                    [int(n_face * 0.9),
                                                     n_face - int(n_face * 0.9)])
    face_ds = MergeDataset(face_cartoon, face_photo_train)

    if stage == 'fit':
      self.ds_train = MultiBatchDataset(scenery_ds, face_ds)
      self.ds_sampler = MultiBatchSampler(
          [MultiRandomSampler(scenery_ds),
           MultiRandomSampler(face_ds)],
          self.sample_steps, self.batch_size)

      self.ds_val = MergeDataset(scenery_photo_val, face_photo_val)
    else:
      self.ds_val = MergeDataset(scenery_photo_val, face_photo_val)

  def train_dataloader(self):
    return DataLoader(self.ds_train,
                      batch_sampler=self.ds_sampler,
                      num_workers=self.num_workers,
                      pin_memory=True)

  def val_dataloader(self):
    return DataLoader(self.ds_val, sampler=MultiRandomSampler(self.ds_val),
                      batch_size=4, num_workers=4)

  def test_dataloader(self):
    return DataLoader(self.ds_val, sampler=MultiRandomSampler(self.ds_val),
                      batch_size=4, num_workers=4)
