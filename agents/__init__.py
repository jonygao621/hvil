from . import acausal
from . import bubblesort
from . import karel_weak
from . import rnn_agent


def catalog(env, config):
    return {
        'Acausal': acausal.AcausalPHPAgent,
        'Bubblesort': bubblesort.BubblesortPHPAgent,
        'Karel-Weak': karel_weak.KarelWeakPHPAgent,
        'RNN': rnn_agent.RNNAgent,
    }[config.name](env, config)
