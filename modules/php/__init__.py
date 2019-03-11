from . import mlp
from . import mnist
from .php import PHP


def catalog(config):
    return {
        'mlp': mlp.MLP,
        'mnist': mnist.MNIST,
    }[config.name](config)
