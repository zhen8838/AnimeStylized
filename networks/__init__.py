from utils.registry import Registry

NETWORKS = Registry('network')
PRETRAINEDS = Registry('pretrained')


def build_network(registry: Registry, name: str, **kwarg: dict):
  if kwarg == None:
    kwarg = {}
  return registry.get(name)(**kwarg)


def get_network(cfg: dict):
  return build_network(NETWORKS, **cfg)

def get_pretrain(cfg: dict):
  return build_network(PRETRAINEDS, **cfg)
