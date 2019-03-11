import gc
import inspect
import math
import os
import re
from collections import abc

# noinspection PyPep8Naming
import torch as T
from torch.nn.utils import rnn as rnn_utils

from . import DictTree
from . import torch_wrapper as torch

CHECKPOINT_PATTERN = re.compile(r'^checkpoint-(\d+)$')


def one_hot(i, n, dtype=torch.float, device='cpu'):
    if isinstance(n, abc.Sequence):
        i = n.index(i)
        n = len(n)
    else:
        assert i < n, f"{i} < {n}"
    res = [0.] * n
    res[i] = 1.
    return torch.tensor(res, dtype=dtype, device=device)


def lookup(x1h, n):
    return n[x1h.argmax()]


def squeeze(t, d=None):
    if t.nelement():
        return torch.squeeze(t, d)
    else:
        return t


def unsqueeze(t, d):
    if t.nelement():
        return torch.unsqueeze(t, d)
    else:
        return t


def getitem(t, *i):
    if t.nelement():
        return t.__getitem__(*i)
    return t


def stack(seq, d=0):
    return torch.cat([unsqueeze(t, d) for t in seq])


def pad(t, l, d):
    if t.shape[d] > l:
        raise ValueError(t.shape, l, d)
    if t.shape[d] < l:
        t = torch.cat([t, t.new_zeros(l - t.shape[d])], d)
    return t


def apply2packed(f, ps):
    return rnn_utils.PackedSequence(f(ps.data), ps.batch_sizes)


def eq(x, y):
    if x is None:
        return y is None
    elif isinstance(x, abc.Mapping):
        if not isinstance(y, abc.Mapping):
            return False
        elif set(x.keys()) != set(y.keys()):
            return False
        else:
            for k in x.keys():
                if not eq(x[k], y[k]):
                    return False
            else:
                return True
    elif isinstance(x, str):
        if not isinstance(y, str):
            return False
        else:
            return x == y
    elif isinstance(x, abc.Sequence):
        if not isinstance(y, abc.Sequence):
            return False
        elif len(x) != len(y):
            return False
        else:
            for i, j in zip(x, y):
                if not eq(i, j):
                    return False
            else:
                return True
    elif isinstance(x, torch.Tensor):
        if not isinstance(y, torch.Tensor):
            return False
        elif x.shape != y.shape:
            return False
        else:
            e = (x == y)  # type: torch.Tensor
            return e.all()
    else:
        raise NotImplementedError


def placed(module=None, *, device=None):
    def _decorate(m):
        class Placed(m):
            async def forward(self, *args, **kwargs):
                if device is None:
                    d = next(self.parameters()).device
                else:
                    d = device
                all_args = DictTree(
                    {f'arg{i}': arg for i, arg in enumerate(args)},
                    kwargs=DictTree(kwargs))
                for k, v in all_args.allitems():
                    if isinstance(v, torch.Tensor):
                        all_args[k] = v.to(d)
                args = [all_args[f'arg{i}'] for i in range(len(args))]
                kwargs = all_args.kwargs
                res = await super().forward(*args, **kwargs)
                for k, v in res.allitems():
                    if isinstance(v, torch.Tensor):
                        res[k] = v.cpu()
                return res

        return Placed

    if module is None:
        return _decorate
    else:
        return _decorate(module)


class Saver(object):
    def __init__(self, agent, config, **components):
        self.agent = agent
        self.components = components
        self.keep_save_freq = config.keep_save_freq
        self.save_dir = config.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save(self, step, ignore=None):
        base_path = os.path.join(self.save_dir, 'checkpoint')
        latest_path = base_path + '-latest'
        step_padded = f'-{step:08}'
        step_path = base_path + step_padded

        agent_state = self.agent.state_dict()
        if ignore:
            for key in agent_state.keys():
                for item in ignore:
                    if key.startswith(item):
                        agent_state.pop(key)

        state = {
            'step': step,
            'agent': agent_state,
            **{k: v.state_dict() for k, v in self.components.items()},
        }
        torch.save(state, step_path)

        # Create symlink
        if os.path.exists(latest_path):
            os.unlink(latest_path)
        # The symlink source is a relative path
        source = 'checkpoint' + step_padded
        os.symlink(source, latest_path)

        # Cull old checkpoints
        if self.keep_save_freq is not None:
            all_checkpoints = []
            for name in os.listdir(self.save_dir):
                if name != source:
                    m = CHECKPOINT_PATTERN.match(name)
                    if m is not None:
                        checkpoint_step = int(m.group(1))
                        all_checkpoints.append((checkpoint_step, name))
            all_checkpoints.sort()

            last_step = -math.inf
            for checkpoint_step, name in all_checkpoints:
                if checkpoint_step - last_step < self.keep_save_freq:
                    os.unlink(os.path.join(self.save_dir, name))
                else:
                    last_step = checkpoint_step

    def restore(self, map_to_cpu=False, step=None):
        path = os.path.join(self.save_dir, 'checkpoint')
        if step is None:
            path += '-latest'
        else:
            path += f'-{step:08}'
        if os.path.exists(path):
            print(f"Loading model from {path}")
            if map_to_cpu:
                checkpoint = torch.load(path, map_location=lambda storage, location: storage)
            else:
                checkpoint = torch.load(path)
            self.agent.load_state_dict(checkpoint['agent'])
            for k, v in self.components.items():
                v.load_state_dict(checkpoint[k])
            return checkpoint['step'], checkpoint['step']
        else:
            return 0, -math.inf


def instrument(owner, func_name, debug=False):
    current = getattr(owner, func_name)
    if current.__name__ == 'my_func':
        return

    def my_func(*args, **kwargs):
        try:
            res = current(*args, **kwargs)
        except RuntimeError:
            print(func_name)
            raise
        previous_frame = inspect.currentframe().f_back
        while previous_frame is not None:
            filename, line_number, function_name, lines, index = inspect.getframeinfo(previous_frame)
            if 'Dynamic' in filename or '/' not in filename:
                break
            previous_frame = previous_frame.f_back
        else:
            previous_frame = inspect.currentframe().f_back
            while previous_frame is not None:
                filename, line_number, function_name, lines, index = inspect.getframeinfo(previous_frame)
                print(f'{filename}:{line_number}')
                previous_frame = previous_frame.f_back
            return
        my_line = f'{filename.split("/")[-1]}:{line_number} ({current.__name__})'
        if debug:
            print(f'instrumented function {owner.__name__}.{func_name} used from {my_line}')
        if isinstance(res, torch.Tensor) and not hasattr(res, '_my_line'):
            setattr(res, '_my_line', my_line)
        return res

    setattr(owner, func_name, my_func)


def instrument_all():
    for func_name in dir(torch.Tensor):
        func = getattr(torch.Tensor, func_name)
        if callable(func) and not isinstance(func, type) and func_name not in ['__init__', '__repr__', '__str__']:
            instrument(torch.Tensor, func_name)
    instrument(T, 'addmm')
    instrument(torch, 'squeeze')
    instrument(torch, 'stack')
    instrument(torch, 'sum')
    instrument(torch, 'unsqueeze')
    instrument(torch, 'where')


def count_tensors(requires_grad=True, instrumented=True):
    return sum(
        1 for t in gc.get_objects()
        if isinstance(t, torch.Tensor)
        and (not requires_grad or t.requires_grad)
        and (not instrumented or hasattr(t, '_my_line')))


def get_instrumented_lines(requires_grad=True):
    # noinspection PyProtectedMember
    return set(
        t._my_line for t in gc.get_objects()
        if isinstance(t, torch.Tensor)
        and (not requires_grad or t.requires_grad)
        and hasattr(t, '_my_line'))


def make_hook(var, name):
    def debug_hook(*args):
        print(f"backpropagating through {id(var)}, {name}, {var.shape}")
        # print(var.data)
        # print(var.shape, torch.tensor(var).numpy())
        return args[0]

    if True and var.requires_grad:
        var.register_hook(debug_hook)


def hook_instrumented(requires_grad=True):
    for t in gc.get_objects():
        if isinstance(t, torch.Tensor) and (not requires_grad or t.requires_grad):
            # noinspection PyProtectedMember
            make_hook(t, t._my_line if hasattr(t, '_my_line') else '?')
