from .commons import PretrainNet
import torchvision
from .gan.mobilefacenet import MobileFaceNet
from .regress import Res18landmarkNet
import torch
from datamodules.dsfunction import denormalize, normalize
import torch.functional as F
import numpy as np
import torch.nn as nn
import torch.nn.functional as nf
from utils.terminfo import ERROR
from typing import List, Tuple, Dict
import types


class StopForward(Exception):
  """The pretrain model forward stop signal
  """
  pass


def named_basic_children(model: nn.Module, prefix='') -> List[Tuple[str, nn.Module]]:
  """Extract model basic module class object

  Args:
      model (nn.Module): model
      prefix (str, optional): Defaults to ''.

  Returns:
      List[Tuple[str, nn.Module]]: [(name,basic_module_object)]
  """
  named_basic = []

  def inner_children(model, prefix=''):
    named_children = list(model.named_children())
    if len(named_children) == 0:
      named_basic.append((prefix, model))
    for name, children in named_children:
      inner_children(children, prefix + ('.' if prefix else '') + name)

  inner_children(model, prefix)
  # rename list for beautiful print
  for i, (name, mod) in enumerate(named_basic):
    s = str(mod.__class__).split('.')[-1].split('\'')[0]
    named_basic[i] = (f'{s}-{i}', mod)
  return named_basic


def featrue_extract_wrapper(model: nn.Module,
                            output_index: int,
                            extract_tuple: bool = False
                            ) -> List[str]:
  """ extract featrue from any basic layer
    NOTE:
      1. This function will modify model forward inplace!
      2. The model must use `_forward_impl` method
      3. When extracted featrue num == 1, model will retrun tensor.
  Args:
      model (nn.Module): main model
      output_index (int): output index int or list[int]
      extract_tuple (bool, optional): weather convert tuple to value. Defaults to False.

  Raises:
      StopForward: stop froward signial

  Returns:
      List[str]: extracted_name list
  """
  names: List[str] = []
  basics: List[nn.Module] = []
  named_basic = named_basic_children(model)
  for name, basic in named_basic:
    names.append(name)
    basics.append(basic)

  del named_basic

  # for multi-featrue outputs
  output_indexs: List[int] = None
  if isinstance(output_index, int):
    output_indexs = [output_index]
  else:
    output_indexs = output_index

  # add hook
  extracted_features = {}
  extracted_num = len(output_indexs)
  extracted_name: List[str] = []
  extracted_count = 0

  for i, idx in enumerate(output_indexs):

    def hook(module: nn.Module, input: torch.Tensor):
      if not extract_tuple:
        input = input[0]
      extracted_features[names[idx]] = input
      if extracted_count == extracted_num:
        raise StopForward

    basics[idx].register_forward_pre_hook(hook)
    extracted_count += 1
    extracted_name.append(names[idx])

  # overwrite model forwar function
  def forward(self, x):
    try:
      y = self._forward_impl(x)
    except StopForward as e:
      if extracted_num == 1:
        return extracted_features[extracted_name[0]]
      else:
        return extracted_features
    return y
  model.forward = types.MethodType(forward, model)
  return extracted_name


class ResNetPreTrained(PretrainNet):
  def __init__(self):
    super().__init__()
    self.res = torchvision.models.resnet18(pretrained=True)

  def _forward_impl(self, x):
    # See note [TorchScript super()]
    x = self.res.conv1(x)
    x = self.res.bn1(x)
    x = self.res.relu(x)
    x = self.res.maxpool(x)

    x = self.res.layer1(x)
    x = self.res.layer2(x)
    x = self.res.layer3(x)
    x = self.res.layer4(x)

    # x = self.avgpool(x)
    # x = torch.flatten(x, 1)
    # x = self.fc(x)
    return x


class VGGPreTrained(PretrainNet):
  def __init__(self):
    """pytorch vgg pretrained net

    Args:
        output_index (int, optional): output layers index. Defaults to 26.
        NOTE the finally output layer name is `output_index-1`
        ```
          (0): Conv2d (1): ReLU
          (2): Conv2d (3): ReLU
          (4): MaxPool2d
          (5): Conv2d (6): ReLU
          (7): Conv2d (8): ReLU
          (9): MaxPool2d
          (10): Conv2d (11): ReLU
          (12): Conv2d (13): ReLU
          (14): Conv2d (15): ReLU
          (16): Conv2d (17): ReLU
          (18): MaxPool2d
          (19): Conv2d (20): ReLU
          (21): Conv2d (22): ReLU
          (23): Conv2d (24): ReLU
          (25): Conv2d (26): ReLU
          (27): MaxPool2d
          (28): Conv2d (29): ReLU
          (30): Conv2d (31): ReLU
          (32): Conv2d (33): ReLU
          (34): Conv2d (35): ReLU
          (36): MaxPool2d
        ```
    """
    super().__init__()
    vgg = torchvision.models.vgg19(pretrained=True)
    self.features = vgg.features
    del vgg

  def _process(self, x):
    # NOTE 图像范围为[-1~1]，先denormalize到0-1再归一化
    return self.vgg_normalize(denormalize(x))

  def setup(self, device: torch.device):
    mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    self.vgg_normalize = lambda x: normalize(x, mean, std)
    self.freeze()

  def _forward_impl(self, x):
    x = self._process(x)
    # See note [TorchScript super()]
    # NOTE get output with out relu activation
    x = self.features(x)
    # x = self.avgpool(x)
    # x = torch.flatten(x, 1)
    # x = self.fc(x)
    return x

  def get_name_list(self):
    named_children = list(self.features.named_children())
    return [name + '_' + mod._get_name() for name, mod in named_children]


class FacePreTrained(PretrainNet):
  def __init__(self, weights_path):
    super().__init__()
    self.model = MobileFaceNet(512)
    try:
      self.model.load_state_dict(torch.load(weights_path))
    except FileNotFoundError as e:
      print(ERROR, "weights_path:", weights_path,
            'does not exits!, if you want to training must download pretrained weights')

  def infer(self, batch_tensor):
    # crop face
    h, w = batch_tensor.shape[2:]
    top = int(h / 2.1 * (0.8 - 0.33))
    bottom = int(h - (h / 2.1 * 0.3))
    size = bottom - top
    left = int(w / 2 - size / 2)
    right = left + size
    batch_tensor = batch_tensor[:, :, top: bottom, left: right]

    batch_tensor = nf.interpolate(
        batch_tensor, size=[112, 112], mode='bilinear', align_corners=True)

    features = self.model(batch_tensor)
    return features

  def cosine_distance(self, batch_tensor1, batch_tensor2):
    feature1 = self.infer(batch_tensor1)
    feature2 = self.infer(batch_tensor2)
    return 1 - torch.cosine_similarity(feature1, feature2)

  def forward(self, batch_tensor1, batch_tensor2):
    return self.cosine_distance(batch_tensor1, batch_tensor2)


class Res18FaceLandmarkPreTrained(PretrainNet):
  def __init__(self, weights_path='models/facelandmark_full.pth'):
    super().__init__()
    self.model = Res18landmarkNet(landmark_num=5, pretrained=False)
    self.model.load_state_dict(torch.load(weights_path))

  def _process(self, x):
    # NOTE 图像范围为[-1~1]，先denormalize到0-1再归一化
    return self.normalize(denormalize(x))

  def setup(self, device: torch.device):
    mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    self.normalize = lambda x: normalize(x, mean, std)
    self.freeze()

  def _forward_impl(self, x):
    x = self._process(x)
    x = self.model(x)
    pred = torch.sigmoid(x)
    # NOTE get output [batch,n_landmark,2]
    return torch.reshape(pred, (-1, 5, 2))


class VGGCaffePreTrained(PretrainNet):
  cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,
         'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

  def __init__(self, weights_path: str = 'models/vgg19.npy') -> None:
    super().__init__()
    try:
      data_dict: dict = np.load(weights_path, encoding='latin1', allow_pickle=True).item()
      self.features = self.make_layers(self.cfg, data_dict)
      del data_dict
    except FileNotFoundError as e:
      print(ERROR, "weights_path:", weights_path,
            'does not exits!, if you want to training must download pretrained weights')

  def _process(self, x):
    # NOTE 图像范围为[-1~1]，先denormalize到0-1再归一化
    rgb = denormalize(x) * 255  # to 255
    bgr = rgb[:, [2, 1, 0], :, :]  # rgb to bgr
    return self.vgg_normalize(bgr)  # vgg norm

  def setup(self, device: torch.device):
    mean: torch.Tensor = torch.tensor([103.939, 116.779, 123.68], device=device)
    mean = mean[None, :, None, None]
    self.vgg_normalize = lambda x: x - mean
    self.freeze()

  def _forward_impl(self, x):
    x = self._process(x)
    # NOTE get output with out relu activation
    x = self.features(x)
    return x

  @staticmethod
  def get_conv_filter(data_dict, name):
    return data_dict[name][0]

  @staticmethod
  def get_bias(data_dict, name):
    return data_dict[name][1]

  @staticmethod
  def get_fc_weight(data_dict, name):
    return data_dict[name][0]

  def make_layers(self, cfg, data_dict, batch_norm=False) -> nn.Sequential:
    layers = []
    in_channels = 3
    block = 1
    number = 1
    for v in cfg:
      if v == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        block += 1
        number = 1
      else:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        with torch.no_grad():
          """ set value """
          weight = torch.FloatTensor(self.get_conv_filter(data_dict, f'conv{block}_{number}'))
          weight = weight.permute((3, 2, 0, 1))
          bias = torch.FloatTensor(self.get_bias(data_dict, f'conv{block}_{number}'))
          conv2d.weight.set_(weight)
          conv2d.bias.set_(bias)
        number += 1
        if batch_norm:
          layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
          layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v

    return nn.Sequential(*layers)

  def get_name_list(self):
    named_children = list(self.features.named_children())
    return [name + '_' + mod._get_name() for name, mod in named_children]
