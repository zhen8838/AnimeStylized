from datamodules.animegands import AnimeGANDataModule
from datamodules.dataset import MergeDataset, MultiRandomSampler, ImageFolder, DataLoader


class FeatrueReconDataModule(AnimeGANDataModule):
  def val_dataloader(self):
    val_root = (self.root / 'test/test_photo256')
    val_cartoon_root = (self.root / 'test/label_map')
    self.ds_val = MergeDataset(
        ImageFolder(val_root.as_posix(), transform=self.val_transform),
        ImageFolder(val_cartoon_root.as_posix(), transform=self.val_transform)
    )

    return DataLoader(
        self.ds_val,
        sampler=MultiRandomSampler(self.ds_val),
        batch_size=4,
        num_workers=4)
