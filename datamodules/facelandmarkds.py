import torchvision
from typing import List, Iterable, Tuple, Dict
import json
import cv2
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datamodules.dataset import DataFolder, random_split, DataLoader
import datamodules.dsfunction as F
import glob
import os
import pytorch_lightning as pl
import torch
LABELIDXMAP = {
    'left_eye': 0,
    'right_eye': 1,
    'nose': 2,
    'left_mouth': 3,
    'right_mouth': 4
}


def vis_keypoints(image, keypoints,
                  color=(0, 255, 0), diameter=15):
  image = image.copy()

  for (x, y) in keypoints:
    cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

  plt.figure(figsize=(8, 8))
  plt.axis('off')
  plt.imshow(image)


class AnnoationSub(object):
  label: str
  points: List[Tuple[float, float]]
  group_id: int
  shape_type: str
  flags: Dict


class AnnoationBase(object):
  version: str
  flags: Dict
  shapes: List[AnnoationSub]
  imagePath: str
  imageData: str
  imageHeight: str
  imageWidth: str


def load_json(json_path: str) -> AnnoationBase:
  with open(json_path, 'r+') as f:
    anno: AnnoationBase = json.load(f)
  return anno


def save_json(anno: AnnoationBase, json_path: str):
  with open(json_path, 'w') as f:
    text = json.dumps(anno, indent=4)
    f.write(text)


class ToTensor(ToTensorV2):
  @property
  def targets(self):
    return {
        "image": self.apply,
        "mask": self.apply_to_mask,
        "keypoints": self.apply_to_keypoints,
    }

  def apply_to_keypoints(self, keypoints, **params):
    return torch.FloatTensor(keypoints)


class Normalize(A.Normalize):
  @property
  def targets(self):
    return {
        "image": self.apply,
        "keypoints": self.apply_to_keypoints,
    }

  def apply_to_keypoints(self, keypoints, **params):
    return keypoints


class FaceLandMarkDataModule(pl.LightningDataModule):
  def __init__(self, root: str, pattern: str,
               im_size: int = 256,
               batch_size: int = 8, num_workers: int = 4,
               augment=True, normalize=True, totenor=True):
    super().__init__()
    self.root = root
    self.pattern = pattern
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.augment = augment
    self.normalize = normalize
    self.totenor = totenor

    fn = lambda x, **kwarg: x
    idenity = A.Lambda(image=fn, mask=fn,
                       keypoint=fn, bbox=fn)

    self.train_transform = A.Compose([
        A.OneOf([
            A.HorizontalFlip(),
            A.ShiftScaleRotate(shift_limit=0.2)
        ]) if augment else idenity,
        A.Resize(im_size, im_size),
        Normalize() if normalize else idenity,
        ToTensor() if totenor else idenity,
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    self.val_transform = A.Compose([
        A.Resize(im_size, im_size),
        Normalize() if normalize else idenity,
        ToTensor() if totenor else idenity,
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

  @staticmethod
  def load_image_and_landmark(json_path: str):
    json_str = load_json(json_path)
    landmark_gt = [0] * 5
    for shape in json_str['shapes']:
      landmark_gt[LABELIDXMAP[shape['label']]] = shape['points'][0]

    im_path = json_path.rsplit('.')[0] + '.jpg'
    image = F.imread(im_path)
    return {'image': image, 'keypoints': landmark_gt}

  def setup(self, stage=None):
    if stage == 'fit':
      train_datafolder = DataFolder(self.root,
                                    self.load_image_and_landmark,
                                    self.pattern, transform=self.train_transform)
      val_datafolder = DataFolder(self.root,
                                  self.load_image_and_landmark,
                                  self.pattern, transform=self.val_transform)
      length = len(train_datafolder)
      self.ds_train, _ = random_split(
          train_datafolder, [length - int(length * 0.1), int(length * 0.1)])
      _, self.ds_val = random_split(
          val_datafolder, [length - int(length * 0.1), int(length * 0.1)])

  def train_dataloader(self):
    return DataLoader(self.ds_train, shuffle=True,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers,
                      pin_memory=True)

  def val_dataloader(self):
    return DataLoader(self.ds_val, shuffle=False,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.ds_val, shuffle=False,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers)
