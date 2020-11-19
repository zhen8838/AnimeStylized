import torch
import torch.functional as F
import torch.nn as nn


def l2_loss(t: torch.Tensor):
  return torch.sum(torch.square(t)) / 2


def huber_loss(y_true, y_pred, delta=1.0):
  """Computes Huber loss value.

  For each value x in `error = y_true - y_pred`:

  ```
  loss = 0.5 * x^2                  if |x| <= d
  loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    delta: A float, the point where the Huber loss function changes from a
      quadratic to linear.

  Returns:
    Tensor with one scalar loss entry per sample.
  """
  error = y_pred - y_true
  abs_error = torch.abs(error)
  quadratic = torch.minimum(abs_error, delta)
  linear = abs_error - quadratic
  return torch.mean(0.5 * quadratic * quadratic + delta * linear)


def variation_loss(image: torch.Tensor, ksize=1):
  """
  A smooth loss in fact. Like the smooth prior in MRF.
  V(y) = || y_{n+1} - y_n ||_2
  """
  dh = image[:, :, :-ksize, :] - image[:, :, ksize:, :]
  dw = image[:, :, :, :-ksize] - image[:, :, :, ksize:]
  return (torch.mean(torch.abs(dh)) + torch.mean(torch.abs(dw)))


def rgb2yuv(rgb: torch.Tensor) -> torch.Tensor:
  """ rgb2yuv NOTE rgb image value range must in 0~1

  Args:
      rgb (torch.Tensor): 4D tensor , [b,c,h,w]

  Returns:
      torch.Tensor: 4D tensor, [b,h,w,c] in [0~1]
  """
  kernel = torch.tensor([[0.299, -0.14714119, 0.61497538],
                         [0.587, -0.28886916, -0.51496512],
                         [0.114, 0.43601035, -0.10001026]],
                        dtype=torch.float32, device=rgb.device)
  rgb = F.tensordot(rgb, kernel, [[1], [0]])
  return rgb
