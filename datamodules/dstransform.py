import datamodules.dsfunction as F
from torchvision.transforms import Normalize, Lambda, Compose
import numpy as np
import cv2
import random


class Add(object):
  """remove datamean
  """

  def __init__(self, mean: list):
    self.mean = mean

  def __call__(self, image: np.ndarray):
    """
    Args:
        image (np.ndarray): cv2 image of size (H, W, C).

    """

    return np.clip(image + self.mean, 0, 255).astype('uint8')

  def __repr__(self):
    return self.__class__.__name__ + '(mean={0})'.format(self.mean)


class RandomHorizontalFlip(object):
  def __init__(self, p=0.5):
    super().__init__()
    self.p = p

  def __call__(self, img: np.ndarray):
    if np.random.rand() < self.p:
      return F.hflip(img)
    return img

  def __repr__(self):
    return self.__class__.__name__ + '(p={})'.format(self.p)


class Grayscale(object):
  """Convert image to grayscale.

  Args:
      num_output_channels (int): (1 or 3) number of channels desired for output image

  Returns:
      PIL Image: Grayscale version of the input.
       - If ``num_output_channels == 1`` : returned image is single channel
       - If ``num_output_channels == 3`` : returned image is 3 channel with r == g == b

  """

  def __init__(self, num_output_channels=1):
    self.num_output_channels = num_output_channels

  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be converted to grayscale.

    Returns:
        PIL Image: Randomly grayscaled image.
    """
    return F.to_grayscale(img, num_output_channels=self.num_output_channels)

  def __repr__(self):
    return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)


class Resize(object):
  """Resize the input cv2 Image to the given size.

  Args:
      size (sequence or int): Desired output size. If size is a sequence like
          (h, w), output size will be matched to this. If size is an int,
          smaller edge of the image will be matched to this number.
          i.e, if height > width, then image will be rescaled to
          (size * height / width, size)
      interpolation (int, optional): Desired interpolation. Default is
          ``PIL.Image.BILINEAR``
  """
  interpolation_dict = {
      'INTER_AREA': cv2.INTER_AREA,
      'INTER_BITS': cv2.INTER_BITS,
      'INTER_BITS2': cv2.INTER_BITS2,
      'INTER_CUBIC': cv2.INTER_CUBIC,
      'INTER_LANCZOS4': cv2.INTER_LANCZOS4,
      'INTER_LINEAR': cv2.INTER_LINEAR,
      'INTER_LINEAR_EXACT': cv2.INTER_LINEAR_EXACT,
      'INTER_MAX': cv2.INTER_MAX,
      'INTER_NEAREST': cv2.INTER_NEAREST,
      'INTER_TAB_SIZE': cv2.INTER_TAB_SIZE,
      'INTER_TAB_SIZE2': cv2.INTER_TAB_SIZE2
  }

  def __init__(self, size, interpolation='INTER_LINEAR'):
    assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
    self.size = size
    self.interpolation = self.interpolation_dict[interpolation]
    self.interpolation_str = interpolation

  def __call__(self, img):
    return F.imresize(img, self.size, self.interpolation)

  def __repr__(self):
    return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation_str)


class RandomCrop(object):
  """Crop the given cv2 Image at a random location.

  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """

  def __init__(self, size):
    if isinstance(size, tuple):
      self.size = (int(size), int(size))
    else:
      self.size = size

  @staticmethod
  def get_params(img, output_size):
    """Get parameters for ``crop`` for a random crop.

    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    w, h = img.shape[:2][::-1]
    th, tw = output_size
    if w == tw and h == th:
      return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw

  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.

    Returns:
        PIL Image: Cropped image.
    """
    i, j, h, w = self.get_params(img, self.size)
    return F.crop(img, i, j, h, w)

  def __repr__(self):
    return self.__class__.__name__ + '(size={0})'.format(self.size)


class ToTensor(object):
  """Convert a ``numpy.ndarray`` to tensor.

  Converts a numpy.ndarray (H x W x C) in the range
  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]

  or if the numpy.ndarray has dtype = np.uint8

  In the other cases, tensors are returned without scaling.
  """

  def __call__(self, pic):
    """
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    return F.to_tensor(pic)

  def __repr__(self):
    return self.__class__.__name__ + '()'
