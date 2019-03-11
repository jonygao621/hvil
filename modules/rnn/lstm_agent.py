import torch
from torch.nn import functional

from modules.php import mlp
from modules.php import mnist
from utils import DictTree
from utils import torch_utils


class LSTMAgent(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre = mnist.MNISTModule.Module(config.pre | DictTree(oput_size=config.features_in_size))
        self.lstm = torch.nn.LSTM(
            input_size=config.features_in_size, bidirectional=False, batch_first=config.batch_first,
            num_layers=config.rnn_layers, hidden_size=config.features_out_size)
        self.post = mlp.make_mlp(config.post | DictTree(iput_size=config.features_out_size))

    def forward(self, iput, mem=None):
        iput = torch_utils.apply2packed(self.pre, iput)
        oput, mem = self.lstm(iput, mem)
        oput = torch_utils.apply2packed(self.post, oput)
        return torch_utils.apply2packed(lambda x: functional.log_softmax(x, 1), oput), mem
