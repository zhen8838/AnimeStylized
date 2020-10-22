import torch
import numpy as np
import cv2
from typing import Sequence, List, Tuple, Union


def normalize(im: Union[np.ndarray, torch.Tensor], mean=0.5, std=0.5):
  return (im - mean) / std


def denormalize(im: Union[np.ndarray, torch.Tensor], mean=0.5, std=0.5):
  return im * std + mean


def hflip(im: np.ndarray):
  return cv2.flip(im, 1)


def to_tensor(im: np.ndarray):
  # handle numpy array
  if im.ndim == 2:
    im = im[:, :, None]

  img = torch.from_numpy(im.transpose((2, 0, 1)))
  # backward compatibility
  if isinstance(img, torch.ByteTensor):
    return img.float().div(255)
  else:
    return img


def to_grayscale(img, num_output_channels=1):
  """Convert image to grayscale version of image.

  Args:
      img (cv2 Image): Image to be converted to grayscale.

  Returns:
      cv2 Image: Grayscale version of the image.
          if num_output_channels = 1 : returned image is single channel

          if num_output_channels = 3 : returned image is 3 channel with r = g = b
  """

  img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), -1)
  if num_output_channels == 3:
    img = np.tile(img, [1, 1, 3])
  else:
    raise ValueError('num_output_channels should be either 1 or 3')

  return img


def imread(path: str):
  return cv2.cvtColor(cv2.imread(path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def imresize(img: np.ndarray, dsize: tuple, interpolation):
  return cv2.resize(img, dsize, interpolation=interpolation)


def crop(img, top, left, height, width):
  """Crop the given cv2 Image.

  Args:
      img (cv2 Image): Image to be cropped. (0,0) denotes the top left corner of the image.
      top (int): Vertical component of the top left corner of the crop box.
      left (int): Horizontal component of the top left corner of the crop box.
      height (int): Height of the crop box.
      width (int): Width of the crop box.

  Returns:
      cv2 Image: Cropped image.
  """
  return img[top:top + height, left:left + width]


def reduce_hw(img_hw: List[int], min_hw: List[int]) -> Tuple[int]:
  im_h, im_w = img_hw
  if im_h <= min_hw[0]:
    im_h = min_hw[0]
  else:
    x = im_h % 32
    im_h = im_h - x

  if im_w < min_hw[1]:
    im_w = min_hw[1]
  else:
    y = im_w % 32
    im_w = im_w - y
  return (im_h, im_w)
