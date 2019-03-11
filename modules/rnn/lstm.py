import torch


class LSTM(torch.nn.LSTM):
    def __init__(self, config):
        super().__init__(
            input_size=config.features_in_size, bidirectional=config.bidirectional, batch_first=config.batch_first,
            num_layers=config.num_layers, hidden_size=config.features_out_size)
