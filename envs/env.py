from utils import DictTree


class Environment(object):
    def __init__(self):
        self.arg_in_size = max(self.term_ret_size, max(action.arg_in_size for action in self.actions.values()))
        self.ret_out_size = max(action.ret_out_size for action in self.actions.values())

    @property
    def init_arg_size(self):
        raise NotImplementedError

    @property
    def term_ret_size(self):
        raise NotImplementedError

    @property
    def obs_size(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    @property
    def actions(self):
        raise NotImplementedError

    async def step(self, act_name, act_arg):
        assert act_arg.shape[0] == self.actions[act_name].arg_in_size
        return await self.actions[act_name](DictTree(env=self, arg=act_arg))


class Action(object):
    @property
    def arg_in_size(self):
        raise NotImplementedError

    @property
    def ret_out_size(self):
        raise NotImplementedError

    async def __call__(self, iput):
        raise NotImplementedError
