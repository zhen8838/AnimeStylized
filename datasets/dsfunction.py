import torch
import numpy as np
from typing import Union


def normalize(im: Union[np.ndarray, torch.Tensor], mean=0.5, std=0.5):
  return (im - mean) / std


def denormalize(im: Union[np.ndarray, torch.Tensor], mean=0.5, std=0.5):
  return im * std + mean
