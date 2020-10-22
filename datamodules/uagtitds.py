import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import ImageFolder, MergeDataset, MultiRandomSampler, MultiSequentialSampler
import datamodules.dstransform as transforms


class UagtitGanDataSet(pl.LightningDataModule):
  def __init__(self, root: str, batch_size: int = 8, num_workers: int = 4):
    super().__init__()
    self.root = root
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256 + 30, 256 + 30)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    self.test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

  def setup(self, stage=None):
    if stage == 'fit':
      self.trainA = ImageFolder(self.root + '/trainA',
                                transform=self.train_transform)
      self.trainB = ImageFolder(self.root + '/trainB',
                                transform=self.train_transform)
      self.ds_train = MergeDataset(self.trainA, self.trainB)
      self.testA = ImageFolder(self.root + '/testA',
                               transform=self.test_transform)
      self.testB = ImageFolder(self.root + '/testB',
                               transform=self.test_transform)
      self.ds_test = MergeDataset(self.testA, self.testB)
    else:
      self.testA = ImageFolder(self.root + '/testA',
                               transform=self.test_transform)
      self.testB = ImageFolder(self.root + '/testB',
                               transform=self.test_transform)
      self.ds_test = MergeDataset(self.testA, self.testB)

  def train_dataloader(self):
    return DataLoader(self.ds_train,
                      sampler=MultiRandomSampler(self.ds_train),
                      batch_size=self.batch_size,
                      num_workers=self.num_workers, pin_memory=True)

  def val_dataloader(self):
    return DataLoader(self.ds_test,
                      sampler=MultiSequentialSampler(self.ds_test),
                      batch_size=4)

  def test_dataloader(self):
    return DataLoader(self.ds_test,
                      sampler=MultiSequentialSampler(self.ds_test),
                      batch_size=4)
