from torch.nn.utils import rnn as rnn_utils

from modules import rnn
from utils import DictTree
from utils import torch_utils
from utils import torch_wrapper as torch
from . import agent

DEFAULT_CONFIG = DictTree(
    rnn=DictTree(
        name='lstm-agent',
        batch_first=True,
        pre=DictTree(
            img_starts=[0],
        ),
        features_in_size=64,
        rnn_layers=4,
        features_out_size=64,
        post=DictTree(
            layers=[DictTree(oput_size=64, activation=torch.nn.ReLU)],
        ),
    ),
)


class RNNAgent(agent.Agent):
    def __init__(self, env, config):
        super().__init__(env, DEFAULT_CONFIG | config)
        if not self.config.teacher:
            rnn_config = DictTree(
                pre=DictTree(iput_size=env.obs_size),
                post=DictTree(oput_size=len(self.act_names)),
            )
            self.act_logits = rnn.catalog(rnn_config | self.config.rnn)
            self._opt = None

    def reset(self, init_arg):
        # TODO: this is needed to roll out the agent
        raise NotImplementedError

    async def forward(self, iput):
        # TODO: this is needed to roll out the agent
        raise NotImplementedError

    def get_loss(self, batch):
        loss = self._get_loss(batch)
        stats = DictTree(
            loss=loss,
            per_step=DictTree(loss=loss),
        )
        return stats

    def evaluate(self, traces):
        error = self._get_loss(traces, True)
        return DictTree(per_trace=DictTree(error=error))

    def _get_loss(self, batch, evaluate=False):
        # TODO: optionally use any available annotations
        packed_obs = rnn_utils.pack_sequence([torch.stack([step.obs for step in trace.data.steps]) for trace in batch])
        # TODO: handle act_arg
        packed_act_logprob, _ = self.act_logits(packed_obs.to(self.device))
        if evaluate:
            packed_true_act_idx = rnn_utils.pack_sequence([
                torch.tensor([self.act_names.index(step.act_name) for step in trace.data.steps], device=self.device)
                for trace in batch])
            packed_error = torch_utils.apply2packed(
                lambda x: x.argmax(1) != packed_true_act_idx.data, packed_act_logprob)
            padded_error, _ = rnn_utils.pad_packed_sequence(packed_error, batch_first=True)
            return padded_error.max(1)[0].long().sum().item()
        else:
            packed_true_act1h = rnn_utils.pack_sequence([
                torch.stack([
                    torch_utils.one_hot(step.act_name, self.act_names, device=self.device)
                    for step in trace.data.steps])
                for trace in batch])
            return -(packed_true_act1h.data * packed_act_logprob.data).sum().cpu()
