import pytorch_lightning as pl
from datamodules.dataset import MergeDataset, MultiRandomSampler, ImageFolder, DataLoader
import datamodules.dstransform as T


class FeatrueReconDataModule(pl.LightningDataModule):
  def __init__(self,
               train_real_root: str,
               train_cartoon_root: str,
               val_real_root: str,
               val_cartoon_root: str,
               batch_size: int = 8, num_workers: int = 4,
               resize=True, normalize=True, totenor=True):
    super().__init__()
    self.train_real_root = train_real_root
    self.train_cartoon_root = train_cartoon_root
    self.val_real_root = val_real_root
    self.val_cartoon_root = val_cartoon_root
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.dims = (3, 256, 256)
    idenity = T.Lambda(lambda x: x)
    self.transform = T.Compose([
        T.Resize((256, 256)) if resize else idenity,
        T.ToTensor() if totenor else idenity,
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) if normalize else idenity)])

  def setup(self, stage: str):
    if stage == 'fit':
      self.ds_train = MergeDataset(
          ImageFolder(self.train_real_root,
                      transform=self.transform),
          ImageFolder(self.train_cartoon_root,
                      transform=self.transform))

      self.ds_val = MergeDataset(
          ImageFolder(self.val_real_root,
                      transform=self.transform),
          ImageFolder(self.val_cartoon_root, transform=self.transform))
    else:
      self.ds_val = MergeDataset(
          ImageFolder(self.val_real_root,
                      transform=self.transform),
          ImageFolder(self.val_cartoon_root, transform=self.transform))

  def train_dataloader(self):
    return DataLoader(
        self.ds_train,
        sampler=MultiRandomSampler(self.ds_train),
        batch_size=self.batch_size,
        num_workers=self.num_workers)

  def val_dataloader(self):
    return DataLoader(
        self.ds_val,
        sampler=MultiRandomSampler(self.ds_val),
        batch_size=4,
        num_workers=4)
