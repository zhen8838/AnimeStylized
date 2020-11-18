import torch
import torch.nn as nn
import torchvision
from networks.commons import Mean
from torchvision.models.resnet import model_urls as res_model_urls, load_state_dict_from_url
from networks import NETWORKS


@NETWORKS.register()
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


@NETWORKS.register()
class Res18landmarkNet(nn.Module):
  def __init__(self, landmark_num: int = 5, pretrained=True):
    super().__init__()
    self.res = torchvision.models.resnet18(pretrained=False, num_classes=landmark_num * 2)

    if pretrained:
      state_dict = load_state_dict_from_url(res_model_urls['resnet18'],
                                            progress=True)
      state_dict.pop('fc.weight')
      state_dict.pop('fc.bias')
      self.res.load_state_dict(state_dict, strict=False)

  def forward(self, x):
    return self.res._forward_impl(x)


@NETWORKS.register()
class AttentionRes18landmarkNet(nn.Module):
  def __init__(self, landmark_num: int = 5, pretrained=True):
    """AttentionRes18landmarkNet
    NOTE this model output heatmap is contrary to what is expected. The **weight of the focus is lower**
    Args:
        landmark_num (int, optional): Defaults to 5.
        pretrained (bool, optional): Defaults to True.
    """
    super().__init__()
    self.res = torchvision.models.resnet18(pretrained=False, num_classes=landmark_num * 2)

    if pretrained:
      state_dict = load_state_dict_from_url(res_model_urls['resnet18'],
                                            progress=True)
      state_dict.pop('fc.weight')
      state_dict.pop('fc.bias')
      self.res.load_state_dict(state_dict, strict=False)

    del self.res.avgpool
    del self.res.fc

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
    self.gap_fc = nn.Linear(128, 1, bias=False)
    self.gmp_fc = nn.Linear(128, 1, bias=False)
    self.conv1x1 = nn.Conv2d(128 * 2, 128, kernel_size=1, stride=1, bias=True)
    self.fc = nn.Linear(512, landmark_num * 2)

  def _forward_impl(self, x):
    x = self.res.conv1(x)
    x = self.res.bn1(x)
    x = self.res.relu(x)  # [128,128]

    x = self.res.maxpool(x)  # [64,64]
    x = self.res.layer1(x)
    x = self.res.layer2(x)

    gap = self.avgpool(x)
    gap_logit = self.gap_fc(torch.flatten(gap, 1))
    gap_out = x * self.gap_fc.weight.unsqueeze(2).unsqueeze(3)
    gmp = self.maxpool(x)
    gmp_logit = self.gmp_fc(torch.flatten(gmp, 1))
    gmp_out = x * self.gmp_fc.weight.unsqueeze(2).unsqueeze(3)
    out = torch.cat([gap_out, gmp_out], 1)
    heatmap = torch.sum(out, dim=1, keepdim=True)

    x = self.conv1x1(out)  # reduce concat output channels
    x = self.res.layer3(x)
    x = self.res.layer4(x)

    x = torch.flatten(self.avgpool(x), 1)
    logit = gap_logit + gmp_logit + self.fc(x)
    return logit, heatmap

  def forward(self, x):
    return self._forward_impl(x)


@NETWORKS.register()
class AttentionRes18landmarkNetV2(nn.Module):
  def __init__(self, landmark_num: int = 5, pretrained=True):
    """AttentionRes18landmarkNet
    NOTE add relu base on AttentionRes18landmarkNet
    Args:
        landmark_num (int, optional): Defaults to 5.
        pretrained (bool, optional): Defaults to True.
    """
    super().__init__()
    self.res = torchvision.models.resnet18(pretrained=False, num_classes=landmark_num * 2)

    if pretrained:
      state_dict = load_state_dict_from_url(res_model_urls['resnet18'],
                                            progress=True)
      state_dict.pop('fc.weight')
      state_dict.pop('fc.bias')
      self.res.load_state_dict(state_dict, strict=False)

    del self.res.avgpool
    del self.res.fc

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
    self.gap_fc = nn.Linear(128, 1, bias=False)
    self.gmp_fc = nn.Linear(128, 1, bias=False)
    self.conv1x1 = nn.Conv2d(128 * 2, 128, kernel_size=1, stride=1, bias=True)
    self.relu = nn.ReLU(True)
    self.fc = nn.Linear(512, landmark_num * 2)

  def _forward_impl(self, x):
    x = self.res.conv1(x)
    x = self.res.bn1(x)
    x = self.res.relu(x)  # [128,128]

    x = self.res.maxpool(x)  # [64,64]
    x = self.res.layer1(x)
    x = self.res.layer2(x)

    gap = self.avgpool(x)
    gap_logit = self.gap_fc(torch.flatten(gap, 1))
    gap_out = x * self.gap_fc.weight.unsqueeze(2).unsqueeze(3)

    gmp = self.maxpool(x)
    gmp_logit = self.gmp_fc(torch.flatten(gmp, 1))
    gmp_out = x * self.gmp_fc.weight.unsqueeze(2).unsqueeze(3)

    out = torch.cat([gap_out, gmp_out], 1)
    heatmap = torch.sum(out, dim=1, keepdim=True)

    x = self.relu(self.conv1x1(out))  # reduce concat output channels
    x = self.res.layer3(x)
    x = self.res.layer4(x)

    x = torch.flatten(self.avgpool(x), 1)
    logit = gap_logit + gmp_logit + self.fc(x)
    return logit, heatmap

  def forward(self, x):
    return self._forward_impl(x)
