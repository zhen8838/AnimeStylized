import types
import copy
import torch.nn as nn


class StopForwardError(RuntimeError):
  pass


class Extractor():
  def __init__(self,
               extract_key: str,
               is_extract_tuple: bool = False,
               is_auto_stop: bool = True) -> None:
    """ ectract any layer output by key

    Args:
        extract_key (str): str or list[str]
        is_extract_tuple (bool, optional): weather return tuple. Defaults to False.
        is_auto_stop (bool, optional): when all output extracted, weather stop forward. Defaults to True.
    """
    super().__init__()
    self.extract_key = extract_key
    self.is_auto_stop = is_auto_stop
    self.is_extract_tuple = is_extract_tuple
    self.hooks = []

  def __call__(self, model: nn.Module):
    return self.inject(model)

  def inject(self, model: nn.Module):
    named_layers = self.get_layers(model)
    # for multi-featrue outputs
    if isinstance(self.extract_key, str):
      output_keys = [self.extract_key]
    else:
      # NOTE remove duplicate key
      seen = set()
      output_keys = [x for x in self.extract_key if not (x in seen or seen.add(x))]

    # add hook
    extracted_features = {}
    extracted_num = len(output_keys)
    extracted_name = []
    extracted_count = 0

    def hook_creator(count: int, key: str):
      def hook(module, input, output):
        extracted_features[
            key] = input if self.is_extract_tuple else input[0]
        if self.is_auto_stop:
          if count == (extracted_num - 1):
            raise StopForwardError

      return hook

    for key in output_keys:
      hook = hook_creator(extracted_count, key)
      self.hooks.append(
          named_layers[key].register_forward_hook(hook))
      extracted_count += 1
      extracted_name.append(key)

    # overwrite model forward function
    if self.is_auto_stop:
      old_forward = copy.copy(model.forward)

      def forward(self, x):
        try:
          y = old_forward(x)
        except StopForwardError as e:
          if extracted_num == 1:
            return extracted_features[extracted_name[0]]
          else:
            return extracted_features
        return y

      model.forward = types.MethodType(forward, model)
    return extracted_name, extracted_features

  @staticmethod
  def get_layers(model: nn.Module):
    return dict(model.named_modules())
