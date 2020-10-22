from torchvision.datasets import VisionDataset
from datamodules.dsfunction import imread
from torch.utils.data import Dataset, RandomSampler, Sampler, DataLoader, TensorDataset
import os
from typing import List
from itertools import cycle, islice
import torch


class ImageFolder(VisionDataset):
  def __init__(self, root, transforms=None, transform=None, target_transform=None):
    super().__init__(root, transforms, transform, target_transform)
    self.loader = imread
    self.samples = os.listdir(root)

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int):
    path = self.samples[index]
    sample = self.loader(self.root + '/' + path)
    if self.transform is not None:
      sample = self.transform(sample)

    return sample

  def size(self, idx):
    return len(self.samples)


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
