from . import lstm
from . import lstm_agent


def catalog(config):
    return {
        'lstm': lstm.LSTM,
        'lstm-agent': lstm_agent.LSTMAgent,
    }[config.name](config)
