from envs import env
from utils import DictTree
from utils import torch_utils
from utils import torch_wrapper as torch


class PHP(object):
    def __init__(self, config):
        self.sub_names = [None] + config.sub_names
        self.subs = None
        self._sub_names_closure = None
        self.arg_in_size = config.arg_in_size
        self.ret_in_size = None
        self.arg_out_size = None
        self.ret_out_size = config.ret_out_size
        self.clocked = config.get('clocked', True)

    def finalize(self, phps, actions, config=None):
        self.subs = {}
        for sub_name in self.sub_names:
            if sub_name is None:
                self.subs[sub_name] = None
            elif sub_name in actions:
                self.subs[sub_name] = actions[sub_name]
            else:
                self.subs[sub_name] = phps[sub_name]
        self.ret_in_size = max(sub.ret_out_size for sub in self.subs.values() if sub is not None)
        self.arg_out_size = max(self.ret_out_size, max(
            sub.arg_in_size for sub in self.subs.values() if sub is not None))

    def sub_names_closure(self):
        if self._sub_names_closure is None:
            self._sub_names_closure = set()
            for sub_name, sub in self.subs.items():
                if sub is None:
                    continue
                elif isinstance(sub, env.Action):
                    self._sub_names_closure.add(sub_name)
                else:
                    self._sub_names_closure |= sub.sub_names_closure()
        return self._sub_names_closure

    async def __call__(self, iput):
        raise NotImplementedError


class PHPModule(torch.nn.Module, PHP):
    def __init__(self, config):
        super().__init__()
        PHP.__init__(self, config)
        self.config = config
        self.sub_module = None
        self.arg_module = None

    def finalize(self, phps, actions, config=None):
        super().finalize(phps, actions)
        config.sub.p_iput_size = self.arg_in_size + self.clocked + len(
            self.sub_names) + self.ret_in_size + self.config.obs_size
        config.sub.q_iput_size = self.arg_in_size + self.clocked + len(
            self.sub_names) + self.ret_in_size + self.config.ctx_size
        config.sub.oput_size = len(self.sub_names)
        self.sub_module = config.sub.module_cls(config.sub)
        if self.arg_out_size:
            config.arg.p_iput_size = config.sub.p_iput_size + len(self.sub_names)
            config.arg.q_iput_size = config.sub.q_iput_size + len(self.sub_names)
            config.arg.oput_size = self.arg_out_size
            self.arg_module = config.arg.module_cls(config.arg)

    async def forward(self, iput):
        """
        iput:
            is_root
            arg
            cnt
            ret = (ret_name, ret_val)
            obs
            ctx                       [optional]
            sub = (sub_name, sub_arg) [optional]
            act = (act_name, act_arg) [optional]
        modes:
            no ctx + no sub + no act = rollout:                sample p(sub | arg, cnt, ret, obs)
            no ctx + no sub +    act = evaluate:               sample p(sub | arg, cnt, ret, obs),
                                                                   and compute error(sub, iput.act)
               ctx +    sub +    act = get_loss (annotated):   compute -log p(sub | arg, cnt, ret, obs)
                                                                      - log q(sub | arg, cnt, ret, ctx, act)
               ctx + no sub +    act = get_loss (unannotated): sample q(sub | arg, cnt, ret, ctx, act),
                                                                   and compute D[q(. | arg, cnt, ret, ctx, act)
                                                                              || p(. | arg, cnt, ret, obs)]
                                                                       and log q(sub | arg, cnt, ret, ctx, act)
        oput:
            sub = (sub_name, sub_arg)
            error                     [in evaluate]
            loss                      [in get_loss]
            log_q                     [in get_loss]
        """
        assert not iput.arg.requires_grad
        assert not iput.cnt.requires_grad
        assert not iput.ret_val.requires_grad
        assert not iput.obs.requires_grad
        has_sub = 'sub_name' in iput
        if has_sub:
            assert not iput.sub_arg.requires_grad
        has_act = 'act_name' in iput
        if has_act:
            assert not iput.act_arg.requires_grad
        iput.ret_val = torch_utils.pad(iput.ret_val, self.ret_in_size, 0)
        iput.ret1h = torch_utils.one_hot(iput.ret_name, self.sub_names, dtype=iput.arg.dtype, device=iput.arg.device)
        if 'ctx' in iput:
            sub_iput = DictTree(
                p_iput=self._get_iput(iput, is_posterior=False, is_arg=False)[None],
                p_mask=self._get_mask(iput, is_posterior=False)[None],
                q_iput=self._get_iput(iput, is_posterior=True, is_arg=False)[None],
                q_mask=self._get_mask(iput, is_posterior=True)[None],
            )
            if has_sub:
                sub1h = torch_utils.one_hot(iput.sub_name, self.sub_names, dtype=iput.arg.dtype, device=iput.arg.device)
                sub_iput.true_oput = sub1h[None]
                sub_oput = await self.sub_module(sub_iput)
                sub_name = iput.sub_name
            else:
                sub_oput = await self.sub_module(sub_iput)
                [sub1h] = sub_oput.oput
                sub_name = torch_utils.lookup(sub1h, self.sub_names)
            [sub_loss] = sub_oput.loss
            [sub_log_p] = sub_oput.log_p
            [sub_log_q] = sub_oput.log_q
            if self.arg_module is None:
                assert self.arg_out_size == 0
                if has_sub:
                    sub_arg = iput.sub_arg
                else:
                    sub_arg = iput.arg.new_empty(0)
                arg_loss = iput.arg.new_zeros(())
                arg_log_p = iput.arg.new_zeros(())
                arg_log_q = iput.arg.new_zeros(())
            else:
                sub_arg_size = self.ret_out_size if sub_name is None else self.subs[sub_name].arg_in_size
                sub_is_arg = self._sub_is_arg(iput | DictTree(sub_name=sub_name))
                arg_iput = DictTree(
                    p_iput=self._get_iput(iput | DictTree(sub1h=sub1h), is_posterior=False, is_arg=True)[None],
                    oput_size=iput.arg.new_full((), sub_arg_size, dtype=torch.int64)[None],
                )
                if sub_is_arg:
                    if has_sub:
                        # noinspection PyUnresolvedReferences
                        assert (iput.sub_arg == iput.act_arg).all()
                    # TODO: use iput.act_arg as auxiliary task
                    arg_iput.true_oput = torch_utils.pad(iput.act_arg, self.arg_out_size, 0)[None]
                else:
                    arg_iput.q_iput = self._get_iput(iput | DictTree(sub1h=sub1h), is_posterior=True, is_arg=True)[None]
                    if has_sub:
                        arg_iput.true_oput = torch_utils.pad(iput.sub_arg, self.arg_out_size, 0)[None]
                arg_oput = await self.arg_module(arg_iput)
                if has_sub:
                    sub_arg = iput.sub_arg
                else:
                    [sub_arg] = arg_oput.oput.detach()[:, :sub_arg_size]
                [arg_loss] = arg_oput.loss
                [arg_log_p] = arg_oput.log_p
                [arg_log_q] = arg_oput.log_q
            oput = DictTree(
                sub_name=sub_name,
                sub_arg=sub_arg,
                loss=[sub_loss, arg_loss],
                log_p=[sub_log_p, arg_log_p],
                log_q=[sub_log_q, arg_log_q],
            )
        else:
            sub_iput = DictTree(
                p_iput=self._get_iput(iput, is_posterior=False, is_arg=False)[None],
                p_mask=self._get_mask(iput, is_posterior=False)[None],
            )
            if has_act and iput.act_name in self.sub_names:
                sub_iput.eval_oput = torch_utils.one_hot(
                    iput.act_name, self.sub_names, dtype=iput.arg.dtype, device=iput.arg.device)[None]
            sub_oput = await self.sub_module(sub_iput)
            [sub1h] = sub_oput.oput
            sub_name = torch_utils.lookup(sub1h, self.sub_names)
            if has_act and iput.act_name in self.sub_names:
                [sub_error] = sub_oput.error.detach()
            else:
                sub_error = True
            if self.arg_module is None:
                assert self.arg_out_size == 0
                sub_arg = iput.arg.new_empty(0)
                arg_error = False
            else:
                sub_arg_size = self.ret_out_size if sub_name is None else self.subs[sub_name].arg_in_size
                arg_iput = DictTree(
                    p_iput=self._get_iput(iput | DictTree(sub1h=sub1h), is_posterior=False, is_arg=True)[None],
                    oput_size=iput.arg.new_full((), sub_arg_size, dtype=torch.int64)[None],
                )
                if has_act and not sub_error:
                    arg_iput.eval_oput = torch_utils.pad(iput.act_arg, self.arg_out_size, 0)[None]
                arg_oput = await self.arg_module(arg_iput)
                [sub_arg] = arg_oput.oput.detach()[:, :sub_arg_size]
                if has_act and not sub_error:
                    [arg_error] = arg_oput.error.detach()
                else:
                    arg_error = True
            oput = DictTree(
                sub_name=sub_name,
                sub_arg=sub_arg,
            )
            if has_act:
                oput.error = bool(sub_error) or bool(arg_error)
        return oput

    def _get_iput(self, iput, is_posterior, is_arg):
        iputs = [iput.arg]
        if self.clocked:
            iputs.append(iput.cnt)
        iputs.extend([iput.ret1h, iput.ret_val, iput.ctx if is_posterior else iput.obs])
        if is_arg:
            iputs.append(iput.sub1h)
        return torch.cat(iputs)

    def _get_mask(self, iput, is_posterior):
        mask = iput.arg.new_ones(len(self.sub_names), dtype=torch.uint8)
        if iput.cnt == 0:  # can't terminate immediately
            mask[self.sub_names.index(None)] = 0
        if is_posterior:
            if iput.act_name is None:
                term_idx = self.sub_names.index(None)
                mask[:term_idx] = 0
                mask[term_idx + 1:] = 0
            else:
                for sub_idx, sub_name in enumerate(self.sub_names):
                    if sub_name is None:
                        if iput.is_root:
                            mask[sub_idx] = 0
                    elif isinstance(self.subs[sub_name], env.Action):
                        if sub_name != iput.act_name:
                            mask[sub_idx] = 0
                    elif iput.act_name not in self.subs[sub_name].sub_names_closure():
                        mask[sub_idx] = 0
        assert mask.sum() > 0
        return mask

    @staticmethod
    def _sub_is_arg(iput):
        # sub_arg must equal the given act_arg if sub_name is the given act_name on which we condition
        #   (except don't confuse non-root termination for a None act_name that indicates root termination)
        return iput.sub_name == iput.act_name and (iput.act_name is not None or iput.is_root)
