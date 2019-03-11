from torch.nn import functional

from utils import DictTree
from utils import torch_wrapper as torch
from . import categorical
from . import php


class MNISTModule(object):
    class Module(torch.nn.Module):
        img_size = 28 * 28

        def __init__(self, config):
            super().__init__()
            self.img_starts = config.img_starts
            # this architecture was adapted from pytorch examples:
            #   https://github.com/pytorch/examples/blob/master/mnist/main.py
            self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
            # fc1 gets iput, but with each img replaced by convnet(img) of len 320
            self.fc1 = torch.nn.Linear(config.iput_size - (self.img_size - 320) * len(self.img_starts), 50)
            self.fc2 = torch.nn.Linear(50, config.oput_size)

        def forward(self, iput):
            features = []
            iputs = [iput]
            for img_start in sorted(self.img_starts, reverse=True):
                iput = iputs.pop()
                img_end = img_start + self.img_size
                img = iput[:, img_start:img_end].view(-1, 1, 28, 28)
                if img_end < iput.shape[1]:
                    iputs.append(iput[:, img_end:])
                if img_start > 0:
                    iputs.append(iput[:, :img_start])
                x = functional.relu(functional.max_pool2d(self.conv1(img), 2))
                x = functional.relu(functional.max_pool2d(self.conv2(x), 2))
                features.append(x.view(-1, 320))
            x = torch.cat(iputs + features, 1)
            x = functional.relu(self.fc1(x))
            return self.fc2(x)

    @staticmethod
    def _make_module(config):
        if config.posterior:
            iput_size = config.q_iput_size
            img_starts = config.q_img_starts
        else:
            iput_size = config.p_iput_size
            img_starts = config.p_img_starts
        return MNISTModule.Module(config | DictTree(iput_size=iput_size, img_starts=img_starts))


class CategoricalMNISTModule(MNISTModule, categorical.CategoricalModule):
    pass


class MNIST(php.PHPModule):
    def finalize(self, phps, actions, config=None):
        module_config = DictTree(
            p_img_starts=self.config.p_img_starts,
            q_img_starts=self.config.q_img_starts,
        )
        super().finalize(phps, actions, (config or DictTree()) | DictTree(
            sub=module_config | DictTree(module_cls=CategoricalMNISTModule),
            arg=module_config | DictTree(module_cls=CategoricalMNISTModule),
        ))
