from . import acausal
from . import bubblesort
from . import karel


def catalog(config):
    return {
        'Acausal': acausal.Acausal,
        'Bubblesort': bubblesort.Bubblesort,
        'Karel': karel.Karel,
    }[config.name](config)
