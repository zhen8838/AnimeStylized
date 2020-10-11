import torch


class DummyOptimizer(torch.optim.Optimizer):
  def __init__(self):
    self.param_groups = []
    self.state = {}

  def zero_grad(self):
    pass

  def step(self, closure):
    pass
