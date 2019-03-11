from utils import DictTree
from utils import torch_wrapper as torch
from . import php_agent
from modules.php import php


class BubblesortPHPAgent(php_agent.PHPAgent):
    root_php_name = 'bubblesort'
    mlp_layers = [DictTree(oput_size=16, activation=torch.nn.ReLU)]

    def _php_configs(self, env):
        return {
                'bubblesort': DictTree(
                    teacher=self.Teacher.Bubblesort,
                    sub_names=['bubble', 'resetptr'],
                    model=DictTree(
                        name='mlp',
                        clocked=True,
                        arg_in_size=1,
                        ret_out_size=0,
                        sub_layers=self.mlp_layers,
                        arg_layers=[],
                    ),
                ),
                'bubble': DictTree(
                    teacher=self.Teacher.Bubble,
                    sub_names=['ptr2right', 'bstep'],
                    model=DictTree(
                        name='mlp',
                        clocked=False,
                        arg_in_size=0,
                        ret_out_size=0,
                        sub_layers=self.mlp_layers,
                        arg_layers=[],
                    ),
                ),
                'resetptr': DictTree(
                    teacher=self.Teacher.Reset,
                    sub_names=['lshift'],
                    model=DictTree(
                        name='mlp',
                        clocked=False,
                        arg_in_size=0,
                        ret_out_size=0,
                        sub_layers=self.mlp_layers,
                        arg_layers=[],
                    ),
                ),
                'bstep': DictTree(
                    teacher=self.Teacher.Bstep,
                    sub_names=['swap', 'rshift'],
                    model=DictTree(
                        name='mlp',
                        clocked=False,
                        arg_in_size=0,
                        ret_out_size=0,
                        sub_layers=self.mlp_layers,
                        arg_layers=[],
                    ),
                ),
                'lshift': DictTree(
                    teacher=self.Teacher.Lshift,
                    sub_names=['ptr1left', 'ptr2left'],
                    model=DictTree(
                        name='mlp',
                        clocked=True,
                        arg_in_size=0,
                        ret_out_size=0,
                        sub_layers=self.mlp_layers,
                        arg_layers=[],
                    ),
                ),
                'rshift': DictTree(
                    teacher=self.Teacher.Rshift,
                    sub_names=['ptr1right', 'ptr2right'],
                    model=DictTree(
                        name='mlp',
                        clocked=True,
                        arg_in_size=0,
                        ret_out_size=0,
                        sub_layers=self.mlp_layers,
                        arg_layers=[],
                    ),
                ),
            }

    posterior_rnn_config = DictTree(
        name='lstm',
        bidirectional=True,
        num_layers=1,
        features_out_size=32,
    )

    class Teacher(object):
        class Bubblesort(php.PHP):
            async def __call__(self, iput):
                if (iput.ret_name in [None, 'resetptr']) and (iput.cnt[0] != (2 * iput.arg - 2)):
                    sub_name = 'bubble'
                elif (iput.ret_name == 'bubble') and (iput.cnt[0] != (2 * iput.arg - 2)):
                    sub_name = 'resetptr'
                elif iput.cnt[0] == (2 * iput.arg - 2):
                    sub_name = None
                sub_arg = torch.empty(0)
                oput = DictTree(
                    sub_name=sub_name,
                    sub_arg=sub_arg,
                )
                return oput

        class Bubble(php.PHP):
            async def __call__(self, iput):
                if (iput.ret_name is None):
                    sub_name = 'ptr2right'
                elif (iput.obs[10] != 1) or (iput.obs[21] != 1) or (iput.ret_name == 'ptr2right'):
                    sub_name = 'bstep'
                elif (iput.obs[10] == 1) and (iput.obs[21] == 1):
                    sub_name = None
                sub_arg = torch.empty(0)
                oput = DictTree(
                    sub_name=sub_name,
                    sub_arg=sub_arg,
                )
                return oput

        class Reset(php.PHP):
            async def __call__(self, iput):
                if (iput.ret_name is None) or (iput.obs[10] != 1) or (iput.obs[21] != 1):
                    sub_name = 'lshift'
                else:
                    sub_name = None
                sub_arg = torch.empty(0)
                oput = DictTree(
                    sub_name=sub_name,
                    sub_arg=sub_arg,
                )
                return oput

        class Bstep(php.PHP):
            async def __call__(self, iput):
                if (int(iput.obs[:10].argmax()) > int(iput.obs[11:21].argmax())) and (iput.ret_name != 'rshift'):
                    sub_name = 'swap'
                elif (int(iput.obs[:10].argmax()) <= int(iput.obs[11:21].argmax())) and (iput.ret_name != 'rshift'):
                    sub_name = 'rshift'
                elif (iput.ret_name == 'rshift'):
                    sub_name = None
                sub_arg = torch.empty(0)
                oput = DictTree(
                    sub_name=sub_name,
                    sub_arg=sub_arg,
                )
                return oput

        class Lshift(php.PHP):
            async def __call__(self, iput):
                sub_name = [None, 'ptr1left', 'ptr2left'][int((iput.cnt[0] + 1) % 3)]
                sub_arg = torch.empty(0)
                oput = DictTree(
                    sub_name=sub_name,
                    sub_arg=sub_arg,
                )
                return oput

        class Rshift(php.PHP):
            async def __call__(self, iput):
                sub_name = [None, 'ptr1right', 'ptr2right'][int((iput.cnt[0] + 1) % 3)]
                sub_arg = torch.empty(0)
                oput = DictTree(
                    sub_name=sub_name,
                    sub_arg=sub_arg,
                )
                return oput