from torch.nn import functional

from utils import DictTree
from utils import batching
from utils import torch_utils
from utils import torch_wrapper as torch

ENTROPY_WEIGHT = 0


@batching.batched
@torch_utils.placed
class CategoricalModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: implement shared weights
        self.p_module = self._make_module(config | DictTree(posterior=False))
        self.q_module = self._make_module(config | DictTree(posterior=True))

    @staticmethod
    def _make_module(config):
        raise NotImplementedError

    async def forward(self, iput):
        """
        iput:
            p = (p_iput, p_mask)
            q = (q_iput, q_mask) [optional]
            oput_size            [optional]
            true_oput            [optional]
            eval_oput            [optional]
        modes:
            no q + no true_oput + no eval_oput = rollout:                sample p(oput | iput, mask)
            no q + no true_oput +    eval_oput = evaluate:               sample p(oput | iput, mask),
                                                                             and compute error(oput, eval_oput)
            no q +    true_oput + no eval_oput = get_loss (act_arg):     compute -log p(oput | iput, mask)
               q +    true_oput + no eval_oput = get_loss (annotated):   compute -log p(oput | iput, mask)
                                                                                - log q(oput | iput, mask)
               q + no true_oput + no eval_oput = get_loss (unannotated): sample q(oput | iput, mask),
                                                                             and compute D[q(. | iput, mask)
                                                                                        || p(. | iput, mask)]
                                                                                 and log q(oput | iput, mask)
        res:
            oput
            error    [in evaluate]
            loss     [in get_loss]
            log_p    [in get_loss]
            log_q    [in get_loss]
        """
        p_log_prob = self._get_log_prob(self.p_module, iput.p_iput, iput.get('p_mask'), iput.get('oput_size'))
        if 'q_iput' in iput:
            q_log_prob = self._get_log_prob(self.q_module, iput.q_iput, iput.get('q_mask'), iput.get('oput_size'))
            oput = iput.get('true_oput', self._sample(q_log_prob))
            log_p = self._log_prob(p_log_prob, oput)
            log_q = self._log_prob(q_log_prob, oput)
            if 'true_oput' in iput:
                loss = -log_p - log_q
            else:
                loss = self._dkl(q_log_prob, p_log_prob)
            res = DictTree(
                oput=oput,
                # TODO: have configurable entropy_weight
                loss=loss + ENTROPY_WEIGHT * (self._neg_entropy(p_log_prob) + self._neg_entropy(q_log_prob)),
                log_p=log_p,
                log_q=log_q,
            )
        else:
            oput = iput.get('true_oput', self._sample(p_log_prob))
            res = DictTree(
                oput=oput,
            )
            if 'true_oput' in iput:
                log_p = self._log_prob(p_log_prob, iput.true_oput)
                res.loss = -log_p + ENTROPY_WEIGHT * self._neg_entropy(p_log_prob)
                res.log_p = log_p
                res.log_q = torch.zeros_like(log_p)
            if 'eval_oput' in iput:
                res.error = self._error(oput, iput.eval_oput)
        return res

    @staticmethod
    def _get_log_prob(module, iput, oput_mask=None, oput_size=None):
        logits = module(iput)
        mask = 1.
        if oput_mask is not None:
            mask *= oput_mask
        if oput_size is not None:
            mask *= (torch.arange(logits.shape[1], dtype=torch.int64, device=logits.device)[None, :] <
                     oput_size[:, None])
        masked_logits = torch.where(mask, logits, logits.new_full((), -1e30))
        return functional.log_softmax(masked_logits, 1)

    @staticmethod
    def _log_prob(log_prob, oput):
        return (oput * log_prob).sum(1)

    @staticmethod
    def _dkl(q_log_prob, p_log_prob):
        return (torch.exp(q_log_prob) * (q_log_prob - p_log_prob)).sum(1)

    def _sample(self, logits):
        if self.training:
            oput = torch.distributions.OneHotCategorical(logits=logits).sample()
        else:
            oput = logits.new_zeros(logits.shape)
            oput.scatter_(1, torch.argmax(logits, 1, True), 1.)
        return oput

    @staticmethod
    def _error(oput, eval_oput):
        return torch.argmax(oput, 1) != torch.argmax(eval_oput, 1)

    @staticmethod
    def _neg_entropy(log_prob):
        return (torch.exp(log_prob) * log_prob).sum(1)
