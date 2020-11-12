from typing import List, Iterable, Tuple, Dict
import json
import cv2
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datamodules.dataset import DataFolder, random_split, DataLoader
import datamodules.dsfunction as F
from pathlib import Path
import os
import pytorch_lightning as pl
import torch
import webdataset as wds
LABELIDXMAP = {
    'left_eye': 0,
    'right_eye': 1,
    'nose': 2,
    'left_mouth': 3,
    'right_mouth': 4
}


LANDMARKS = ['left_eye', 'right_eye',
             'nose', 'left_mouth',
             'right_mouth']


def get_pattern_and_total_num(root, stage='train') -> Tuple[str, int]:
  root = Path(root)
  splits = []
  patten = ''
  name = ''
  total = ''
  for s in list(root.glob(f'{stage}*')):
    name, total, split = s.stem.split('-')
    splits.append(split)
  if len(splits) > 1:
    patten_str = '{' + '..'.join([splits[0], splits[-1]]) + '}'
  else:
    patten_str = splits[0]
  patten = (root / ('-'.join([name, total, patten_str]) + '.tar')).as_posix()
  return patten, int(total)


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


def get_face_annotation(xmin: float, ymin: float,
                        xmax: float, ymax: float,
                        group_id: int) -> AnnoationSub:
  return {'label': 'face',
          'points': [[xmin.item(), ymin.item()],
                     [xmax.item(), ymax.item()]],
          'group_id': group_id,
          'shape_type': 'rectangle',
          'flags': {}}


def get_landmark_annotation(x: float, y: float,
                            label: str,
                            group_id: int) -> AnnoationSub:
  return {'label': label,
          'points': [[x, y]],
          'group_id': group_id,
          'shape_type': 'point',
          'flags': {}}


def get_base_annotation(path: str, Height: int, Width: int) -> AnnoationBase:
  return {'version': '4.5.6',
          'flags': {},
          'shapes': [],
          'imagePath': path,
          'imageData': None,
          'imageHeight': Height,
          'imageWidth': Width}


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
               use_webdataset: bool = False,
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
    self.use_webdataset = use_webdataset

    fn = lambda x, **kwarg: x
    idenity = A.Lambda(image=fn, mask=fn,
                       keypoint=fn, bbox=fn)

    self.train_transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2) if augment else idenity,
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
  def parser_landmark(json_str: str):
    landmark_gt = [0] * 5
    for shape in json_str['shapes']:
      landmark_gt[LABELIDXMAP[shape['label']]] = shape['points'][0]
    return landmark_gt

  @staticmethod
  def load_image_and_landmark(json_path: str):
    json_str = load_json(json_path)
    landmark_gt = FaceLandMarkDataModule.parser_landmark(json_str)
    im_path = json_path.rsplit('.')[0] + '.jpg'
    image = F.imread(im_path)
    return {'image': image, 'keypoints': landmark_gt}

  def setup(self, stage=None):
    if stage == 'fit':
      if not self.use_webdataset:
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
      else:

        def train_perar_fn(sample):
          keypoints = FaceLandMarkDataModule.parser_landmark(sample['json'])
          return self.train_transform(image=sample['jpg'], keypoints=keypoints)

        def test_perar_fn(sample):
          keypoints = FaceLandMarkDataModule.parser_landmark(sample['json'])
          return self.val_transform(image=sample['jpg'], keypoints=keypoints)

        train_root_pattern, train_total = get_pattern_and_total_num(self.root, 'train')
        self.ds_train = (wds.Dataset(train_root_pattern, length=train_total).
                         shuffle(5000).
                         decode('rgb8').
                         map(train_perar_fn))

        test_root_pattern, test_total = get_pattern_and_total_num(self.root, 'test')
        self.ds_val = (wds.Dataset(test_root_pattern, length=test_total).
                       shuffle(5000).
                       decode('rgb8').
                       map(test_perar_fn))

  def train_dataloader(self):
    return DataLoader(self.ds_train, shuffle=False if self.use_webdataset else True,
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
