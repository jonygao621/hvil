import itertools
import collections

import torch

from modules.php import php
from envs.karel_lib import parser_for_synthesis
from utils import DictTree
from . import php_agent

DEFAULT_CONFIG = DictTree(
    levels=3,
    branch=5,
    override_rollout=True,
)


class KarelWeakPHPAgent(php_agent.PHPAgent):
    root_php_name = 'P'
    mlp_layers = [DictTree(oput_size=256, activation=torch.nn.ReLU)]

    def __init__(self, env, config):
        super().__init__(env, DEFAULT_CONFIG | config)

    def _php_configs(self, env):
        act_names = list(env.actions.keys())
        call_graph = collections.defaultdict(list)

        # Fill in call graph for all non-primitive actions
        for level in range(self.config.levels):
            for branches in itertools.product(range(self.config.branch),
                    repeat=level + 1):
                parent = ''.join(str(x) for x in branches[:-1])
                child = ''.join(str(x) for x in branches)
                call_graph[f'P{parent}'].append(f'P{child}')

        # Add primitive actions to call graph
        for level in range(self.config.levels + 1):
            for branches in itertools.product(range(self.config.branch),
                    repeat=level):
                parent = ''.join(str(x) for x in branches)
                call_graph[f'P{parent}'] += act_names
        
        return {
            k: DictTree(
                teacher=self.make_teacher(env),
                sub_names=v,
                model=DictTree(
                    name='mlp',
                    arg_in_size=0,
                    ret_out_size=0,
                    sub_layers=self.mlp_layers,
                    arg_layers=self.mlp_layers,
                ),
            )
            for k, v in call_graph.items()
        }

    posterior_rnn_config = DictTree(
        name='lstm',
        bidirectional=True,
        features_out_size=32,
    )

    # Overriding definition in agent.Agent
    def rollout(self, env):
        if not self.config.override_rollout:
            return super().rollout(env)

        init_arg = env.reset()
        memory = self.reset(init_arg)
        trace = DictTree(
            metadata=DictTree(init_arg=self._get_value(init_arg, teacher=False)),
            data=DictTree(steps=[]),
        )
        ret_name = None
        empty_tensor = torch.empty(0)

        # Run the code
        def pre_action_callback(act_name, metadata):
            nonlocal memory, ret_name

            obs = env.observe()
            top = memory.stack[-1]
            trace.data.steps.append(DictTree(
                mem_in=memory,
                ret_name=ret_name,
                ret_val=empty_tensor,
                obs=self._get_value(obs, teacher=False),
                mem_out=DictTree(
                    steps=[DictTree(
                        name=top.name,
                        arg=self._get_value(top.arg, teacher=False),
                        cnt=top.cnt,
                        ret_name=ret_name,
                        ret_val=empty_tensor,
                        sub_name=act_name,
                        sub_arg=empty_tensor,
                    )],
                    stack=memory.stack[:-1] + [DictTree(top, cnt=top.cnt + 1)]),
                act_name=act_name,
                act_arg=empty_tensor,
            ))

            memory = trace.data.steps[-1].mem_out
            ret_name = act_name

        old_pre_action_callback = env.kr.pre_action_callback
        env.kr.pre_action_callback = pre_action_callback
        env.code()
        env.kr.pre_action_callback = old_pre_action_callback

        top = memory.stack[-1]
        trace.data.steps.append(DictTree(
            mem_in=memory,
            ret_name=ret_name,
            ret_val=empty_tensor,
            obs=self._get_value(env.observe(), teacher=False),
            mem_out=DictTree(
                steps=[DictTree(
                    name=top.name,
                    arg=self._get_value(top.arg, teacher=False),
                    cnt=top.cnt,
                    ret_name=ret_name,
                    ret_val=empty_tensor,
                    sub_name=None,
                    sub_arg=empty_tensor,
                    )],
                stack=[]),
            act_name=None,
            act_arg=empty_tensor,
        ))

        self._postprocess(trace)
        return trace
    
    class DummyTeacher(php.PHP):
        pass

    @staticmethod
    def make_teacher(env):
        class P(php.PHP):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._actions_i = 0

            async def __call__(self, iput):
                sub_name = env.actions_list[self._actions_i]
                result = DictTree(
                        sub_name=sub_name,
                        sub_arg=iput.arg)

                if sub_name is None:
                    self._actions_i = 0
                else:
                    self._actions_i += 1
                return result
        return P

