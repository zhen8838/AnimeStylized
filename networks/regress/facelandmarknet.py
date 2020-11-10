import torch
import torch.nn as nn
import torchvision
from networks.commons import Mean
from torchvision.models.resnet import model_urls as res_model_urls, load_state_dict_from_url


class VGGlandmarkNet(nn.Module):
  def __init__(self, landmark_num: int = 5):
    super().__init__()
    vgg = torchvision.models.vgg19(pretrained=True)
    self.features = vgg.features
    del vgg
    self.landmark_num = landmark_num
    self.regress = nn.Sequential(
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, self.landmark_num * 2),
    )

  def _forward_impl(self, x):
    x = self.features(x)
    x = self.regress(x)
    return x

  def forward(self, x):
    return self._forward_impl(x)


class Res18landmarkNet(nn.Module):
  def __init__(self, landmark_num: int = 5, pretrained=True, pretrained_path: str = None):
    super().__init__()
    self.res = torchvision.models.resnet18(pretrained=False, num_classes=landmark_num * 2)

    if pretrained:
      if pretrained_path:
        state_dict = torch.load(pretrained_path)
        self.load_state_dict(state_dict, strict=True)
      else:
        state_dict = load_state_dict_from_url(res_model_urls['resnet18'],
                                              progress=True)
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        self.res.load_state_dict(state_dict, strict=False)

  def forward(self, x):
    return self.res._forward_impl(x)
