import os
import sys
sys.path.insert(0, os.getcwd())
import pytorch_lightning as pl
import torch
import torch.nn as nn
from scripts.common import run_common, log_images
from networks.regress import Res18landmarkNet
from datamodules.facelandmarkds import FaceLandMarkDataModule


class FaceLandMark(pl.LightningModule):
  def __init__(self, lr: float = 2e-4, im_size: int = 256, landmark_num: int = 5):
    super().__init__()
    self.save_hyperparameters()

    self.net = Res18landmarkNet(landmark_num)
    self.bce = nn.BCEWithLogitsLoss()

  def forward(self, image):
    pred = torch.sigmoid(self.net(image))
    return torch.reshape(pred, (-1, 5, 2))

  def training_step(self, batch, batch_idx):
    image = batch['image']
    landmark_gt = batch['keypoints']
    landmark_gt = torch.flatten(landmark_gt, 1, -1)
    landmark_pred = self.net(image)
    loss = self.bce(landmark_pred, torch.clamp(landmark_gt / self.hparams.im_size, 0, 1))
    self.log('loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    image = batch['image']
    landmark_gt = batch['keypoints']
    landmark_gt = torch.flatten(landmark_gt, 1, -1)
    landmark_pred = self.net(image)
    loss = self.bce(landmark_pred, torch.clamp(landmark_gt / self.hparams.im_size, 0, 1))
    self.log('val_loss', loss)
    self.log('hp_metric', loss)
    return loss

  def configure_optimizers(self):
    opt = torch.optim.Adam(self.net.parameters(),
                           lr=self.hparams.lr,
                           betas=(0.5, 0.999))
    return opt


def infer_fn(model: FaceLandMark, image_path: str):
  from datamodules.dsfunction import imread
  import albumentations as A
  from albumentations.pytorch import ToTensorV2
  from datamodules.facelandmarkds import vis_keypoints
  import matplotlib.pyplot as plt
  model.eval()
  image = imread(image_path)
  orig_height = image.shape[0]
  transform = A.Compose([
      A.Resize(256, 256),
      A.Normalize(),
      ToTensorV2()
  ])
  transformed = transform(image=image)
  trans_image = transformed['image']
  trans_image = torch.unsqueeze(trans_image, 0)
  landmark_pred = model.forward(trans_image)
  landmark_pred = torch.squeeze(landmark_pred, 0)
  landmark_pred = landmark_pred.detach().numpy() * orig_height
  vis_keypoints(image, landmark_pred, diameter=10)
  plt.show()


def export_fn(model: FaceLandMark, save_path: str = 'facelandmark.pt', type: str = 'torchscipt'):
  """ export facelandmark model

  Args:
      model (FaceLandMark): model
      save_path (str): save path
      type (str): export type, choice in ['torchscipt','torch']
  """
  from utils.terminfo import INFO
  if type == 'torch':
    torch.save(model.net.state_dict(), save_path)
  elif type == 'torchscipt':
    model.to_torchscript(save_path, example_inputs=torch.rand(1, 3, 256, 256))

  print(INFO, f"Save model to {save_path}")
  print(INFO, f"Export type is {type}")


if __name__ == "__main__":
  run_common(FaceLandMark, FaceLandMarkDataModule, infer_fn, export_fn)
