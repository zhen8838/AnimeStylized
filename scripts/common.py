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
from datamodules.dsfunction import denormalize


def log_images(cls: pl.LightningModule,
               images_dict: Dict[str, torch.Tensor],
               num: int = 4):
  for k, images in images_dict.items():
    image_show = torchvision.utils.make_grid(images[:num], nrow=num)
    image_show = denormalize(image_show)  # to [0~1]
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


def run_train(model_class, datamodule_clss):
  with open(sys.argv[1]) as f:
    config: dict = safe_load(f)
  datamodule = datamodule_clss(**config['dataset'])
  model: pl.LightningModule = model_class(**config['model'])
  if 'load_from_checkpoint' in config.keys():
    if config['load_from_checkpoint'] is not None:
      # NOTE overwrite the old model hparam, need reset whole module
      model = model.load_from_checkpoint(
          config['load_from_checkpoint'],
          strict=False, **config['model'])

  ckpt_callback = CusModelCheckpoint(**config['checkpoint'])
  logger = TensorBoardLogger(**config['logger'])
  trainer = pl.Trainer(checkpoint_callback=ckpt_callback,
                       logger=logger,
                       **config['trainer'])
  trainer.fit(model, datamodule)
