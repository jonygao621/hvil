from typing import Tuple

import numpy as np
import torchvision as tv

from utils import DictTree
from utils import torch_utils
from utils import torch_wrapper as torch
from . import env


class Acausal(env.Environment):
    img_size = 28 * 28
    num_cats = 10
    num_acts = 2

    def __init__(self, _config):
        super().__init__()
        # the following magic numbers were copied from pytorch examples:
        #   https://github.com/pytorch/examples/blob/master/mnist/main.py
        self.mnist = [tv.datasets.MNIST('data/mnist', download=True, train=train, transform=tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307,), (0.3081,))
        ])) for train in [False, True]]
        self.training = None
        self.digit = None

    @property
    def init_arg_size(self):
        return 0

    @property
    def term_ret_size(self):
        return 0

    @property
    def obs_size(self):
        return self.img_size + self.num_cats

    def reset(self):
        self.digit = None
        return DictTree(value=torch.empty(0))

    def observe(self):
        if self.digit is None:
            self.digit = self.mnist[self.training][
                np.random.choice(len(self.mnist[self.training]))]  # type: Tuple[torch.Tensor, int]
            return DictTree(
                value=torch.cat([self.digit[0].view(self.img_size), torch.zeros(self.num_cats)]),
                teacher_value=torch_utils.one_hot(self.digit[1], 2 * self.num_cats),
            )
        else:
            return DictTree(
                value=torch.cat([torch.zeros(self.img_size), torch_utils.one_hot(self.digit[1], self.num_cats)]),
                teacher_value=torch_utils.one_hot(self.num_cats + self.digit[1], 2 * self.num_cats),
            )

    class Action(env.Action):
        arg_in_size = 0
        ret_out_size = 0

        async def __call__(self, iput):
            return torch.empty(0)

    @property
    def actions(self):
        return {f'A{i}': self.Action() for i in range(self.num_acts)}
