from modules import php
from utils import DictTree
from utils import torch_wrapper as torch
from . import php_agent


class AcausalPHPAgent(php_agent.PHPAgent):
    root_php_name = 'Acausal'
    mlp_layers = [DictTree(oput_size=64, activation=torch.nn.ReLU)]

    def _php_configs(self, env):
        return {
            'Acausal': DictTree(
                teacher=self.Teacher.Acausal,
                sub_names=[f'P{i}' for i in range(5)],
                model=DictTree(
                    name='mnist',
                    arg_in_size=0,
                    ret_out_size=0,
                    # iput: arg (0), cnt (1), ret1h (6), ret_val (0), obs (img_size + num_cats)
                    p_img_starts=[7],
                    # iput: arg (0), cnt (1), ret1h (6), ret_val (0), ctx (features_out_size * num_directions)
                    q_img_starts=[],
                ),
            ),
            'P0': DictTree(
                teacher=self.Teacher.P0,
                sub_names=list(env.actions.keys()),
                model=DictTree(
                    name='mnist',
                    arg_in_size=0,
                    ret_out_size=0,
                    # iput: arg (0), cnt (1), ret1h (3), ret_val (0), obs (img_size + num_cats)
                    p_img_starts=[4],
                    # iput: arg (0), cnt (1), ret1h (3), ret_val (0), ctx (features_out_size * num_directions)
                    q_img_starts=[],
                ),
            ),
            'P1': DictTree(
                teacher=self.Teacher.P1,
                sub_names=list(env.actions.keys()),
                model=DictTree(
                    name='mnist',
                    arg_in_size=0,
                    ret_out_size=0,
                    # iput: arg (0), cnt (1), ret1h (3), ret_val (0), obs (img_size + num_cats)
                    p_img_starts=[4],
                    # iput: arg (0), cnt (1), ret1h (3), ret_val (0), ctx (features_out_size * num_directions)
                    q_img_starts=[],
                ),
            ),
            'P2': DictTree(
                teacher=self.Teacher.P2,
                sub_names=list(env.actions.keys()),
                model=DictTree(
                    name='mnist',
                    arg_in_size=0,
                    ret_out_size=0,
                    # iput: arg (0), cnt (1), ret1h (3), ret_val (0), obs (img_size + num_cats)
                    p_img_starts=[4],
                    # iput: arg (0), cnt (1), ret1h (3), ret_val (0), ctx (features_out_size * num_directions)
                    q_img_starts=[],
                ),
            ),
            'P3': DictTree(
                teacher=self.Teacher.P3,
                sub_names=list(env.actions.keys()),
                model=DictTree(
                    name='mnist',
                    arg_in_size=0,
                    ret_out_size=0,
                    # iput: arg (0), cnt (1), ret1h (3), ret_val (0), obs (img_size + num_cats)
                    p_img_starts=[4],
                    # iput: arg (0), cnt (1), ret1h (3), ret_val (0), ctx (features_out_size * num_directions)
                    q_img_starts=[],
                ),
            ),
            'P4': DictTree(
                teacher=self.Teacher.P4,
                sub_names=list(env.actions.keys()),
                model=DictTree(
                    name='mnist',
                    arg_in_size=0,
                    ret_out_size=0,
                    # iput: arg (0), cnt (1), ret1h (3), ret_val (0), obs (img_size + num_cats)
                    p_img_starts=[4],
                    # iput: arg (0), cnt (1), ret1h (3), ret_val (0), ctx (features_out_size * num_directions)
                    q_img_starts=[],
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
        class Acausal(php.PHP):
            async def __call__(self, iput):
                digit = iput.obs.argmax()
                if iput.cnt == 0:
                    if digit % 2 == 0:
                        sub_name = 'P0'
                    else:
                        sub_name = 'P1'
                else:
                    sub_name = None
                sub_arg = torch.empty(0)
                oput = DictTree(
                    sub_name=sub_name,
                    sub_arg=sub_arg,
                )
                return oput

        class P0(php.PHP):
            async def __call__(self, iput):
                if iput.cnt == 0:
                    # p = [0.01] * 2
                    # p[0] = 0.99
                    # a = np.random.choice(range(2), p=p)
                    a = 0
                    sub_name = f'A{a}'
                else:
                    sub_name = None
                sub_arg = torch.empty(0)
                oput = DictTree(
                    sub_name=sub_name,
                    sub_arg=sub_arg,
                )
                return oput

        class P1(php.PHP):
            async def __call__(self, iput):
                if iput.cnt == 0:
                    # p = [0.01] * 2
                    # p[1] = 0.99
                    # a = np.random.choice(range(2), p=p)
                    a = 1
                    sub_name = f'A{a}'
                else:
                    sub_name = None
                sub_arg = torch.empty(0)
                oput = DictTree(
                    sub_name=sub_name,
                    sub_arg=sub_arg,
                )
                return oput

        class P2(php.PHP):
            async def __call__(self, iput):
                raise NotImplementedError

        class P3(php.PHP):
            async def __call__(self, iput):
                raise NotImplementedError

        class P4(php.PHP):
            async def __call__(self, iput):
                raise NotImplementedError
