import pytorch_lightning as pl
from pathlib import Path
import datamodules.dstransform as transforms
from datamodules.dataset import ImageFolder, MergeDataset, MultiRandomSampler, DataLoader, TensorDataset


class AnimeGANDataModule(pl.LightningDataModule):
  def __init__(self, root: str, style: str,
               batch_size: int = 8, num_workers: int = 4,
               data_mean=[-4.4346957, -8.665916, 13.100612],
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

    self.train_real_transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if augment else idenity,
        transforms.ToTensor() if totenor else idenity,
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) if normalize else idenity])

    self.train_anime_transform = transforms.Compose([
        transforms.Add(data_mean),
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
                               transform=self.train_real_transform)
      train_anime = ImageFolder(anime_root.as_posix(),
                                transform=self.train_anime_transform)
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
