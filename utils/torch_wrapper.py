import torch

# dummy module to suppress warnings
# should be removed when this issue is resolved: https://github.com/pytorch/pytorch/issues/7318

argmax = torch.argmax
cat = torch.cat
distributions = torch.distributions
empty = torch.empty
exp = torch.exp
float = torch.float
full = torch.full
int64 = torch.int64
load = torch.load
min = torch.min
nn = torch.nn
no_grad = torch.no_grad
ones_like = torch.ones_like
optim = torch.optim
save = torch.save
squeeze = torch.squeeze
stack = torch.stack
Tensor = torch.Tensor
uint8 = torch.uint8
unsqueeze = torch.unsqueeze
where = torch.where
zeros_like = torch.zeros_like


def arange(*args, **kwargs):
    return torch.arange(*args, **kwargs)


def ones(*args, **kwargs):
    return torch.ones(*args, **kwargs)


def sum(*args, **kwargs):
    return torch.sum(*args, **kwargs)


def tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs)


def zeros(*args, **kwargs):
    return torch.zeros(*args, **kwargs)
