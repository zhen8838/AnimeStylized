import pytorch_lightning as pl
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler
from torchvision import transforms
import os
import torch
from typing import List
from itertools import cycle, islice


class ImageFolder(VisionDataset):
  def __init__(self, root, transforms=None, transform=None, target_transform=None):
    super().__init__(root, transforms, transform, target_transform)
    self.loader = default_loader
    self.samples = os.listdir(root)

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int):
    path = self.samples[index]
    sample = self.loader(self.root + '/' + path)
    if self.transform is not None:
      sample = self.transform(sample)

    return sample


class MergeDataset(Dataset):
  r"""Dataset wrapping tensors.

  Each sample will be retrieved by indexing tensors along the first dimension.

  Arguments:
      *tensors (Tensor): tensors that have the same size of the first dimension.
  """

  def __init__(self, *tensors):
    self.tensors = tensors
    self.sizes = [len(tensor) for tensor in tensors]

  def __getitem__(self, indexs: List[int]):
    return tuple(tensor[idx] for idx, tensor in zip(indexs, self.tensors))

  def __len__(self):
    return max(self.sizes)


class MultiRandomSampler(RandomSampler):
  def __init__(self, data_source: MergeDataset, replacement=True, num_samples=None, generator=None):
    self.data_source: MergeDataset = data_source
    self.replacement = replacement
    self._num_samples = num_samples
    self.generator = generator
    self.maxn = len(self.data_source)

  @property
  def num_samples(self):
    # dataset size might change at runtime
    if self._num_samples is None:
      self._num_samples = self.data_source.sizes
    return self._num_samples

  def __iter__(self):
    rands = []
    for size in self.num_samples:
      if self.maxn == size:
        rands.append(torch.randperm(size, generator=self.generator).tolist())
      else:
        rands.append(torch.randint(high=size, size=(self.maxn,),
                                   dtype=torch.int64, generator=self.generator).tolist())
    return zip(*rands)

  def __len__(self):
    return len(self.data_source)


class MultiSequentialSampler(Sampler):
  r"""Samples elements sequentially, always in the same order.

  Arguments:
      data_source (Dataset): dataset to sample from
  """

  def __init__(self, data_source: MergeDataset):
    self.data_source: MergeDataset = data_source
    self.num_samples = data_source.sizes
    self.maxn = len(data_source)

  def __iter__(self):
    ls = []
    for size in self.num_samples:
      if self.maxn == size:
        ls.append(range(size))
      else:
        ls.append(islice(cycle(range(size)), self.maxn))
    return zip(*ls)

  def __len__(self):
    return len(self.data_source)


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
