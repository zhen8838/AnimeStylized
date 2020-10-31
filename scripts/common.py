from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from yaml import safe_load
import sys
from typing import Optional
import torch
import numpy as np
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn, rank_zero_info
from typing import Dict
import torchvision
import argparse


def log_images(cls: pl.LightningModule,
               images_dict: Dict[str, torch.Tensor],
               num: int = 4):
  for k, images in images_dict.items():
    image_show = torchvision.utils.make_grid(images[:num], nrow=4, normalize=True)  # to [0~1]
    cls.logger.experiment.add_image(k, image_show, cls.global_step)


class CusModelCheckpoint(ModelCheckpoint):

  def __init_monitor_mode(self, monitor, mode):
    torch_inf = torch.tensor(np.Inf)
    mode_dict = {
        "min": (torch_inf, "min"),
        "max": (-torch_inf, "max"),
        "auto": (-torch_inf, "max")
        if monitor is not None and ("acc" in monitor or monitor.startswith("fmeasure"))
        else (torch_inf, "min"),
        "all": (torch.tensor(0), "all")
    }

    if mode not in mode_dict:
      rank_zero_warn(
          f"ModelCheckpoint mode {mode} is unknown, fallback to auto mode",
          RuntimeWarning,
      )
      mode = "auto"

    self.kth_value, self.mode = mode_dict[mode]

  def check_monitor_top_k(self, current) -> bool:
    if current is None:
      return False

    if self.save_top_k == -1:
      return True

    less_than_k_models = len(self.best_k_models) < self.save_top_k
    if less_than_k_models:
      return True

    if not isinstance(current, torch.Tensor):
      rank_zero_warn(
          f"{current} is supposed to be a `torch.Tensor`. Saving checkpoint may not work correctly."
          f" HINT: check the value of {self.monitor} in your validation loop",
          RuntimeWarning,
      )
      current = torch.tensor(current)

    monitor_op = {"min": torch.lt,
                  "max": torch.gt,
                  "all": torch.tensor(True)}[self.mode]
    return monitor_op(current, self.best_k_models[self.kth_best_model_path]).item()

  def on_validation_end(self, trainer, pl_module):
    """ do not save when after validation"""
    pass

  def on_train_end(self, trainer, pl_module):
    self.save_checkpoint(trainer, pl_module)


def parser_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, help='config file path')
  parser.add_argument('--stage', type=str, help='trian or test or others',
                      choices=['fit', 'test', 'infer'])
  parser.add_argument('--ckpt', type=str, help='pretrained checkpoint file', default=None)
  parser.add_argument('--extra', type=str,
                      help='extra kwargs for infer, must be `a:b,c:d`', default=None)
  return parser.parse_args()


def run_common(model_class: pl.LightningModule,
               datamodule_class: pl.LightningDataModule,
               infer_fn: callable = lambda x, arg: print("do nothing~")):
  args = parser_args()
  model: pl.LightningModule = None
  """ build model """
  if args.stage in ['fit', 'test']:
    config: dict = safe_load(args.config)
    if args.ckpt:
      print("Load from checkpoint", args.ckpt)
      model = model_class.load_from_checkpoint(args.ckpt, strict=False, **config['model'])
    else:
      model = model_class(**config['model'])
    datamodule = datamodule_class(**config['dataset'])
    ckpt_callback = CusModelCheckpoint(**config['checkpoint'])
    logger = TensorBoardLogger(**config['logger'])
    trainer = pl.Trainer(checkpoint_callback=ckpt_callback,
                         logger=logger,
                         **config['trainer'])
    if args.stage == 'fit':
      trainer.fit(model, datamodule)
    elif args.stage == 'test':
      trainer.test(model, datamodule)
  elif args.stage == 'infer':
    print("Load from checkpoint", args.ckpt)
    model = model_class.load_from_checkpoint(args.ckpt, strict=False)
    kwargs = {}
    for line in args.extra.split(','):
      k, v = line.split(':')
      kwargs.setdefault(k, v)
    infer_fn(model, **kwargs)


if __name__ == "__main__":
  args = parser_args()
  print(args)
