import asyncio

from modules import opt
from utils import DictTree
from utils import torch_wrapper as torch

DEFAULT_OPT_CONFIG = DictTree(
    name='adam',
    kwargs=DictTree(
        weight_decay=1e-3,
    ),
    clip_grad_norm=None,
)


class Agent(torch.nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.config = config
        self.config.opt = DEFAULT_OPT_CONFIG | self.config.get('opt', DictTree())
        self.act_names = [None] + list(env.actions.keys())
        self.act_arg_size = {None: env.term_ret_size}
        self.act_arg_size.update({k: v.arg_in_size for k, v in env.actions.items()})
        self.max_act_arg_size = env.arg_in_size
        assert self.max_act_arg_size == max(self.act_arg_size.values())
        self.max_act_ret_size = env.ret_out_size
        self._opt = None

    @property
    def opt(self):
        if self._opt is None:
            if list(self.parameters()):
                self._opt = opt.catalog(self.parameters(), self.config.opt)
        return self._opt

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return 'cpu'

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self.opt is not None:
            state['_opt'] = self.opt.state_dict()
        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict = state_dict.copy()
        if self.opt is not None:
            self.opt.load_state_dict(state_dict.pop('_opt'))
        super().load_state_dict(state_dict, *args, **kwargs)

    def _get_value(self, x, teacher=None):
        if isinstance(x, DictTree):
            if teacher is None:
                teacher = self.config.teacher
            if teacher:
                return x.get('teacher_value', x.value)
            else:
                return x.value
        else:
            return x

    def reset(self, init_arg):
        raise NotImplementedError

    async def forward(self, iput):
        raise NotImplementedError

    def rollout(self, env):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            init_arg = env.reset()
            memory = self.reset(init_arg)
            trace = DictTree(
                metadata=DictTree(init_arg=self._get_value(init_arg, teacher=False)),
                data=DictTree(steps=[]),
            )
            ret_name = None
            ret_val = torch.empty(0)
            done = False
            while not done:
                obs = env.observe()
                iput = DictTree(
                    mem_in=memory,
                    ret_name=ret_name,
                    ret_val=ret_val,
                    obs=obs,
                )
                oput = asyncio.get_event_loop().run_until_complete(self(iput))
                trace.data.steps.append(DictTree(
                    mem_in=memory,
                    ret_name=ret_name,
                    ret_val=self._get_value(ret_val, teacher=False),
                    obs=self._get_value(obs, teacher=False),
                    mem_out=oput.mem_out,
                    act_name=oput.act_name,
                    act_arg=oput.act_arg,
                ))
                if oput.act_name is None:
                    done = True
                else:
                    memory = oput.mem_out
                    ret_name = oput.act_name
                    ret_val = asyncio.get_event_loop().run_until_complete(env.step(oput.act_name, oput.act_arg))
            self._postprocess(trace)
        self.train()  # set to training mode
        return trace

    def _postprocess(self, trace):
        """

        Args:
            trace (DictTree)
        """
        trace.metadata.length = len(trace.data.steps)

    def get_loss(self, batch):
        raise NotImplementedError

    def opt_step(self, loss):
        self.opt.zero_grad()
        loss.backward()
        if self.config.opt.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.opt.clip_grad_norm)
        self.opt.step()

    def evaluate(self, traces):
        raise NotImplementedError
