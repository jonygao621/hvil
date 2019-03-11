import torch


def catalog(params, config):
    return {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
    }[config.name](params, **config.kwargs)
