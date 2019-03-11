import asyncio

from torch.nn.utils import rnn as rnn_utils

from modules import rnn
from utils import DictTree
from utils import torch_utils
from utils import torch_wrapper as torch
from . import agent


class VIAgent(agent.Agent):
    def __init__(self, env, config):
        super().__init__(env, config)
        if not self.config.teacher:
            observable_size = len(self.act_names) + env.ret_out_size + env.obs_size + len(
                self.act_names) + env.arg_in_size
            self.ctx = rnn.catalog(
                DictTree(features_in_size=observable_size, batch_first=True) | self.posterior_rnn_config)
            self._opt = None

    def reset(self, init_arg):
        raise NotImplementedError

    async def forward(self, iput):
        """
        iput:
            mem_in
            ret = (ret_name, ret_val)
            obs
            ctx                       [optional]
            act = (act_name, act_arg) [optional]
            mem_out                   [optional]
        modes:
            no ctx + no act + no mem_out = rollout:                sample p(mem_out, act | mem_in, ret, obs)
            no ctx +    act + no mem_out = evaluate:               sample p(mem_out, act | mem_in, ret, obs),
                                                                       and compute error(act, iput.act)
               ctx +    act +    mem_out = get_loss (annotated):   compute -log p(mem_out, act | mem_in, ret, obs)
                                                                          - log q(mem_out | mem_in, ret, ctx, act)
               ctx +    act + no mem_out = get_loss (unannotated): compute D[log q(mem_out | mem_in, ret, ctx, act)
                                                                          || log p(mem_out, act | mem_in, ret, obs)],
                                                                       and sample q(mem_out | mem_in, ret, ctx, act)
        oput:
            mem_out
            act = (act_name, act_arg) [in rollout]
            error                     [in evaluate]
            loss                      [in get_loss]
            log_q                     [in get_loss (unannotated)]
        """
        raise NotImplementedError

    def _postprocess(self, trace):
        super()._postprocess(trace)

        def _process_step(step):
            return DictTree(
                ret1h=torch_utils.one_hot(
                    step.ret_name, self.act_names, dtype=step.ret_val.dtype, device=step.ret_val.device),
                ret_val=torch_utils.pad(step.ret_val, self.max_act_ret_size, 0),
                obs=step.obs,
                act1h=torch_utils.one_hot(
                    step.act_name, self.act_names, dtype=step.act_arg.dtype, device=step.act_arg.device),
                act_arg=torch_utils.pad(step.act_arg, self.max_act_arg_size, 0),
            )

        steps = [_process_step(step) for step in trace.data.steps]
        trace_data = [torch_utils.stack([step[k] for step in steps]) for k in steps[0].keys()]
        trace.data.all = torch.cat(trace_data, 1)

    def get_loss(self, batch):
        packed_batch = rnn_utils.pack_sequence([trace.data.all for trace in batch])
        packed_ctx, _ = self.ctx(packed_batch.to(self.device))
        padded_ctx, _ = rnn_utils.pad_packed_sequence(packed_ctx.cpu(), batch_first=True)
        get_loss = [self._get_loss(trace | DictTree(ctx=ctx)) for trace, ctx in zip(batch, padded_ctx)]
        all_stats = asyncio.get_event_loop().run_until_complete(asyncio.gather(*get_loss))
        stats = DictTree(
            loss=torch.stack([s.loss for s in all_stats]).sum(),
            per_step=DictTree(),
        )
        with torch.no_grad():
            for k, v in all_stats[0].per_step.allitems():
                stats.per_step[k] = torch.stack([s.per_step[k] for s in all_stats]).sum()
        return stats

    async def _get_loss(self, trace):
        # TODO: time stats
        loss = []
        log_p = []
        log_q = []
        if trace.metadata.annotated:
            memory = trace.data.steps[0].mem_in
        else:
            memory = self.reset(DictTree(value=trace.metadata.init_arg))
        for step, ctx in zip(trace.data.steps, trace.ctx):
            iput = DictTree(
                mem_in=memory,
                ret_name=step.ret_name,
                ret_val=step.ret_val,
                obs=step.obs,
                ctx=ctx,
                act_name=step.act_name,
                act_arg=step.act_arg,
            )
            if trace.metadata.annotated:
                iput.mem_out = step.mem_out
            oput = await self(iput)
            loss.extend(oput.loss)
            log_p.extend(oput.log_p)
            log_q.extend(oput.log_q)
            if not trace.metadata.annotated:
                step.mem_in = memory
                step.mem_out = oput.mem_out
            memory = oput.mem_out
        if trace.metadata.annotated:
            loss = torch.stack(loss).sum()
            with torch.no_grad():
                return DictTree(
                    per_step=DictTree(
                        score=loss,
                        log_p=torch.stack(log_p).sum(),
                        log_q=torch.stack(log_q).sum(),
                    ),
                    loss=loss,
                )
        else:
            # score-function trick
            loss = torch.stack(loss)
            score = loss.sum()
            log_q = torch.stack(log_q)
            loss = score + (loss.detach()[1:] * log_q[:-1].cumsum(0)).sum()
            with torch.no_grad():
                return DictTree(
                    per_step=DictTree(
                        score=score,
                        log_p=torch.stack(log_p).sum(),
                        log_q=log_q.sum(),
                    ),
                    loss=loss,
                )

    def evaluate(self, traces):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            error = asyncio.get_event_loop().run_until_complete(
                asyncio.gather(*[self._error(trace) for trace in traces]))
        self.train()  # set to training mode
        return DictTree(per_trace=DictTree(error=sum(error)))

    async def _error(self, trace):
        memory = self.reset(DictTree(value=trace.metadata.init_arg))
        for step in trace.data.steps:
            iput = DictTree(
                mem_in=memory,
                ret_name=step.ret_name,
                ret_val=step.ret_val,
                obs=step.obs,
                act_name=step.act_name,
                act_arg=step.act_arg,
            )
            oput = await self(iput)
            if oput.error:
                error = True
                break
            memory = oput.mem_out
        else:
            error = False
        return error
