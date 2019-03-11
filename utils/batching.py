import asyncio
import queue
import threading

from utils import DictTree
from . import torch_wrapper as torch


def batched(module):
    class Batched(module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._batch_owner_lock = threading.Lock()
            self._batch_executor_lock = asyncio.Lock()
            self._batch_queue = queue.Queue()

        async def forward(self, *args, **kwargs):
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            job = DictTree(
                meta=DictTree(
                    future=future,
                    num_args=len(args),
                ),
                args=DictTree(
                    {f'arg{i}': arg for i, arg in enumerate(args)},
                    kwargs=DictTree(kwargs)),
            )
            for v in job.args.allvalues():
                if isinstance(v, torch.Tensor):
                    iput_size = v.shape[0]
                    break
            else:
                raise ValueError("batched execution must take at least one tensor argument or keyword argument")
            assert all(v.shape[0] == iput_size for v in job.args.allvalues())
            job.meta.iput_size = iput_size
            self._batch_queue.put(job)
            if self._batch_owner_lock.acquire(False):
                try:
                    try:
                        await self._batch_executor_lock.acquire()
                    finally:
                        self._batch_owner_lock.release()
                    next_batch = []
                    while True:
                        batch = next_batch
                        next_batch = []
                        while not self._batch_queue.empty():
                            new_job = self._batch_queue.get()
                            if len(batch) > 0:
                                old_job = batch[0]
                                compatible = (
                                        {tuple(k) for k in old_job.args.allkeys()} ==
                                        {tuple(k) for k in new_job.args.allkeys()}
                                        # keyword inputs may contain non-tensor values,
                                        #  which must be the same for all jobs
                                        and all((isinstance(old_job.args[k], torch.Tensor) and
                                                 isinstance(v, torch.Tensor)) or old_job.args[k] == v
                                                for k, v in new_job.args.allitems()))
                            else:
                                compatible = True
                            if compatible:
                                batch.append(new_job)
                            else:
                                next_batch.append(new_job)
                                break
                        if len(batch):
                            batch_args, batch_kwargs = batchify(batch)
                            batch_oputs = await super().forward(*batch_args, **batch_kwargs)
                            oputs = unbatchify(batch, batch_oputs)
                            for job, oput in zip(batch, oputs):
                                job.meta.future.set_result(oput)
                        if len(next_batch) == 0 and self._batch_queue.empty():
                            break
                finally:
                    try:
                        self._batch_executor_lock.release()
                    except RuntimeError:
                        pass
            return await future

    return Batched


def batchify(batch):
    batch_args = DictTree()
    for k, v in batch[0].args.allitems():
        if isinstance(v, torch.Tensor):
            batch_args[k] = torch.cat([job.args[k] for job in batch])
        else:
            batch_args[k] = v
    return [batch_args[f'arg{i}'] for i in range(batch[0].meta.num_args)], batch_args.get('kwargs', {})


def unbatchify(batch, batch_oputs):
    oputs = []
    i = 0
    for job in batch:
        iput_size = job.meta.iput_size
        oput = DictTree()
        for k, v in batch_oputs.allitems():
            if isinstance(v, torch.Tensor):
                oput[k] = v[i:i + iput_size]
            else:
                oput[k] = v
        oputs.append(oput)
        i += iput_size
    return oputs
