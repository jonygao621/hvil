import torch

from utils import DictTree
from . import categorical
from . import php


def make_mlp(config):
    """

    Args:
        config (DictTree)
    """
    layers = []
    iput_size = config.iput_size
    for layer in config.layers:
        layers.append(torch.nn.Linear(iput_size, layer.oput_size))
        layers.append(layer.activation())
        iput_size = layer.oput_size
    layers.append(torch.nn.Linear(iput_size, config.oput_size))
    return torch.nn.Sequential(*layers)


class MLPModule(object):
    @staticmethod
    def _make_module(config):
        cfg = DictTree(
            layers=config.layers,
            oput_size=config.oput_size,
        )
        if config.posterior:
            cfg.iput_size = config.q_iput_size
        else:
            cfg.iput_size = config.p_iput_size
        return make_mlp(cfg)


class CategoricalMLPModule(MLPModule, categorical.CategoricalModule):
    pass


class MLP(php.PHPModule):
    def finalize(self, phps, actions, config=None):
        super().finalize(phps, actions, (config or DictTree()) | DictTree(
            sub=DictTree(
                module_cls=CategoricalMLPModule,
                layers=self.config.sub_layers,
            ),
            arg=DictTree(
                module_cls=CategoricalMLPModule,
                layers=self.config.arg_layers,
            ),
        ))
