import torch
import torch.nn.functional as F
from .mobilefacenet import MobileFaceNet
import pytorch_lightning as pl
from collections import OrderedDict


class FaceFeatures(pl.LightningModule):
  def __init__(self, weights_path):
    super().__init__()
    self.model = MobileFaceNet(512)
    self.model.load_state_dict(torch.load(weights_path))

  def setup(self, device: torch.device):
    self.freeze()

  def train(self, mode: bool):
    """ avoid pytorch light auto set trian mode """
    return super().train(False)

  def state_dict(self, destination, prefix, keep_vars):
    """ avoid pytorch light auto save params """
    destination = OrderedDict()
    destination._metadata = OrderedDict()
    return destination

  def infer(self, batch_tensor):
    # crop face
    h, w = batch_tensor.shape[2:]
    top = int(h / 2.1 * (0.8 - 0.33))
    bottom = int(h - (h / 2.1 * 0.3))
    size = bottom - top
    left = int(w / 2 - size / 2)
    right = left + size
    batch_tensor = batch_tensor[:, :, top: bottom, left: right]

    batch_tensor = F.interpolate(batch_tensor, size=[112, 112], mode='bilinear', align_corners=True)

    features = self.model(batch_tensor)
    return features

  def cosine_distance(self, batch_tensor1, batch_tensor2):
    feature1 = self.infer(batch_tensor1)
    feature2 = self.infer(batch_tensor2)
    return 1 - torch.cosine_similarity(feature1, feature2)
