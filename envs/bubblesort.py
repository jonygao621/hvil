import numpy as np

from . import env
from utils import DictTree
from utils import torch_wrapper as torch
from utils import torch_utils
DEFAULT_CONFIG = DictTree(
    num_actions=2,
    agent=DictTree(
        mlp_layers=[DictTree(oput_size=16, activation=torch.nn.ReLU)]
    ),
)

class Bubblesort(env.Environment):
    def __init__(self, config):
        """
        Elevator environment.

        :param config: environment configuration
        """
        super().__init__()
        config = DEFAULT_CONFIG | config
        self._actions = None
        self.max_length = 9
        self.num_actions = 5
        self.pointer1=1
        self.pointer2=1
        self.length = np.random.choice(self.max_length - 1) + 2
        self.list_to_sort = list(np.random.randint(0, 10, self.length))
        self.last_obs = [self.list_to_sort[0], 1, self.list_to_sort[0], 1]


    class ptr1right(env.Action):
        arg_in_size = 0
        ret_out_size = 0

        async def __call__(self, iput):
            env = iput.env
            env.pointer1 = min(env.pointer1 + 1, env.length)
            num1 = torch_utils.one_hot(env.list_to_sort[env.pointer1-1], 10)
            num2 = torch_utils.one_hot(env.list_to_sort[env.pointer2-1], 10)
            env.last_obs = [num1, torch.tensor([float((env.pointer1 == 1) or (env.pointer1 == env.length))]), num2, torch.tensor([float((env.pointer2 == 1) or (env.pointer2 == env.length))])]
            return torch.empty(0)

    class ptr1left(env.Action):
        arg_in_size = 0
        ret_out_size = 0    
        
        async def __call__(self, iput):
            env = iput.env
            env.pointer1 = max(env.pointer1 - 1, 1)
            num1 = torch_utils.one_hot(env.list_to_sort[env.pointer1-1], 10)
            num2 = torch_utils.one_hot(env.list_to_sort[env.pointer2-1], 10)
            env.last_obs = [num1, torch.tensor([float((env.pointer1 == 1) or (env.pointer1 == env.length))]), num2, torch.tensor([float((env.pointer2 == 1) or (env.pointer2 == env.length))])]
            return torch.empty(0)

    class ptr2right(env.Action):
        arg_in_size = 0
        ret_out_size = 0
        
        async def __call__(self, iput):
            env = iput.env
            env.pointer2 = min(env.pointer2 + 1, env.length)
            num1 = torch_utils.one_hot(env.list_to_sort[env.pointer1-1], 10)
            num2 = torch_utils.one_hot(env.list_to_sort[env.pointer2-1], 10)
            env.last_obs = [num1, torch.tensor([float((env.pointer1 == 1) or (env.pointer1 == env.length))]), num2, torch.tensor([float((env.pointer2 == 1) or (env.pointer2 == env.length))])]
            return torch.empty(0)

    class ptr2left(env.Action):
        arg_in_size = 0
        ret_out_size = 0

        async def __call__(self, iput):
            env = iput.env
            env.pointer2 = max(env.pointer2 - 1, 1)
            num1 = torch_utils.one_hot(env.list_to_sort[env.pointer1-1], 10)
            num2 = torch_utils.one_hot(env.list_to_sort[env.pointer2-1], 10)
            env.last_obs = [num1, torch.tensor([float((env.pointer1 == 1) or (env.pointer1 == env.length))]), num2, torch.tensor([float((env.pointer2 == 1) or (env.pointer2 == env.length))])]
            return torch.empty(0)

    class swap(env.Action):
        arg_in_size = 0
        ret_out_size = 0

        async def __call__(self, iput):
            env = iput.env
            temp = env.list_to_sort[env.pointer2-1]
            env.list_to_sort[env.pointer2-1] = env.list_to_sort[env.pointer1-1]
            env.list_to_sort[env.pointer1-1] = temp
            num1 = torch_utils.one_hot(env.list_to_sort[env.pointer1-1], 10)
            num2 = torch_utils.one_hot(env.list_to_sort[env.pointer2-1], 10)
            env.last_obs = [num1, torch.tensor([float((env.pointer1 == 1) or (env.pointer1 == env.length))]), num2, torch.tensor([float((env.pointer2 == 1) or (env.pointer2 == env.length))])]
            return torch.empty(0)

    actions = {action.__name__: action() for action in [ptr1left, ptr1right, ptr2left, ptr2right, swap]}

    @property
    def init_arg_size(self):
        return 1

    @property
    def obs_size(self):
        return 22

    @property
    def term_ret_size(self):
        return 0

    @property
    def act_size(self):
        return self.num_actions

    def reset(self, expert=False):
        """
        Reset environment.

        :return: root argument, one-hot target floor
        """
        self.length = np.random.choice(self.max_length - 1) + 2
        self.list_to_sort = list(np.random.randint(0, 10, self.length))
        self.pointer1=1
        self.pointer2=1
        root_arg = torch.Tensor([self.length])
        num1 = torch_utils.one_hot(self.list_to_sort[self.pointer1-1], 10)
        num2 = torch_utils.one_hot(self.list_to_sort[self.pointer2-1], 10)
        self.last_obs = [num1, torch.tensor([1.]), num2, torch.tensor([1.])]
        return DictTree(value=root_arg, 
                        expert_value=root_arg)

    def observe(self, expert=False):
        """
        Observe environment.

        :return: one-hot current floor
        """
        obs = torch.cat(self.last_obs)
        return DictTree(value=obs, 
                        expert_value=obs)