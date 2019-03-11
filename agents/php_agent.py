from modules import php
from utils import DictTree
from utils import torch_utils
from utils import torch_wrapper as torch
from . import vi_agent


class PHPAgent(vi_agent.VIAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self._make_phps(env)

    def _make_phps(self, env):
        php_configs = self._php_configs(env)
        ctx_size = self.posterior_rnn_config.features_out_size * (1 + self.posterior_rnn_config.bidirectional)
        self.phps = {}
        for name, config in php_configs.items():
            if self.config.teacher:
                php_maker = config.get('teacher', php.PHP)
            else:
                php_maker = php.catalog
            self.phps[name] = php_maker(config.model | DictTree(
                sub_names=config.sub_names,
                obs_size=env.obs_size,
                ctx_size=ctx_size,
            ))
        for name, p in self.phps.items():
            p.finalize(self.phps, env.actions)
            if not self.config.teacher:
                self.add_module(name, p)

    def reset(self, init_arg):
        return DictTree(stack=[DictTree(
            name=self.root_php_name,
            arg=init_arg,
            cnt=torch.zeros(1, device=self._get_value(init_arg).device)),
        ])

    async def forward(self, iput):
        stack = iput.mem_in.stack.copy()
        ret_name = iput.ret_name
        ret_val = iput.ret_val
        steps = []
        loss = []
        log_p = []
        log_q = []
        step_idx = 0
        while True:
            top = stack[-1]
            top_php = self.phps[top.name]
            step_iput = DictTree(
                is_root=(len(stack) == 1),
                arg=self._get_value(top.arg),
                cnt=top.cnt,
                ret_name=ret_name,
                ret_val=self._get_value(ret_val),
                obs=self._get_value(iput.obs),
            )
            if 'ctx' in iput:
                step_iput.ctx = iput.ctx
                if 'mem_out' in iput:
                    step = iput.mem_out.steps[step_idx]
                    step_iput.sub_name = step.sub_name
                    step_iput.sub_arg = step.sub_arg
                    step_idx += 1
                step_iput.act_name = iput.act_name
                step_iput.act_arg = iput.act_arg
            elif 'act_name' in iput:
                step_iput.act_name = iput.act_name
                step_iput.act_arg = iput.act_arg
            step_oput = await top_php(step_iput)
            steps.append(DictTree(
                name=top.name,
                arg=self._get_value(top.arg, teacher=False),
                cnt=top.cnt,
                ret_name=ret_name,
                ret_val=self._get_value(ret_val, teacher=False),
                sub_name=step_oput.sub_name,
                sub_arg=step_oput.sub_arg,
            ))
            if 'ctx' in iput:
                loss.extend(step_oput.loss)
                log_p.extend(step_oput.log_p)
                log_q.extend(step_oput.log_q)
            if step_oput.sub_name is None:
                # terminate php
                assert top.cnt > 0
                stack.pop()
                if stack:
                    ret_name = top.name
                    ret_val = step_oput.sub_arg
                else:
                    # terminate agent
                    act_name = None
                    act_arg = step_oput.sub_arg
                    break
            elif step_oput.sub_name in self.act_names:
                # take action
                stack[-1] = DictTree(top, cnt=top.cnt + 1)
                act_name = step_oput.sub_name
                act_arg = step_oput.sub_arg
                break
            else:
                # call php
                stack[-1] = DictTree(top, cnt=top.cnt + 1)
                ret_name = None
                ret_val = ret_val.new_empty(0)
                stack.append(DictTree(name=step_oput.sub_name, arg=step_oput.sub_arg, cnt=ret_val.new_zeros(1)))
        oput = DictTree(
            mem_out=DictTree(steps=steps, stack=stack),
        )
        if 'mem_out' in iput:
            assert torch_utils.eq(iput.mem_out, oput.mem_out)
        if 'ctx' in iput:
            oput.loss = loss
            oput.log_p = log_p
            oput.log_q = log_q
        elif 'act_name' in iput:
            oput.error = step_oput.error
        else:
            oput.act_name = act_name
            oput.act_arg = act_arg
        return oput
