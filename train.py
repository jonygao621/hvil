import argparse
import contextlib
import itertools
import math
import multiprocessing
import os
from datetime import datetime

import numpy as np
import torch

import agents
import data
import envs
import utils
from utils import DictTree
from utils import tf_utils
from utils import torch_utils

DEFAULT_CONFIG = DictTree(
    trial_datetime=None,
    seed=None,
    logdir='logdir',
    ranges=None,
    num_processes=1,
    num_threads=1,
    device='cuda',
    data=DictTree(
        train=DictTree(
            generate=False,
            load=True,
        ),
        test=DictTree(
            generate=False,
            load=True,
            num_unannotated=0,
            batch_size=None,
            unannotated_throttling_steps=None,
        ),
    ),
    num_steps=10,
    eval_freq=1000,
    save_freq=1000,
    keep_save_freq=1000,
)


def train(config):
    base_config = config
    if config.train.seed is not None:
        np.random.seed(config.train.seed)
        torch.manual_seed(np.random.randint(np.iinfo(np.uint64).max))
    base_logdir = config.train.logdir
    ranges = [config.train[field.split('.')] for field in config.train.ranges or []]
    configs = []
    for values in itertools.product(*ranges):
        if values:
            config = base_config.copy()
            for field, value in zip(config.train.ranges, values):
                config.train[field.split('.')] = value
            if not data.get_num_traces(config.train.data.train):
                continue
            values_str = "; ".join(f"{field}={value}" for field, value in zip(config.train.ranges, values)).replace(
                "/", ".")
            config.train.logdir = os.path.join(base_logdir, config.train.name, values_str, config.train.trial_datetime)
            print(f"Scheduling trainer with config: {values_str}")
        else:
            config.train.logdir = os.path.join(base_logdir, config.train.name, config.train.trial_datetime)
        configs.append(config)
    num_processes = config.train.num_processes or len(configs)
    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as p:
            p.map(_train, configs)
    else:
        [_train(config) for config in configs]


def _train(config):
    torch.set_num_threads(config.train.num_threads)
    print(config.train.logdir)
    env = envs.catalog(config.env)
    agent = agents.catalog(env, DictTree(teacher=False) | config.agent)
    agent.to(config.train.device)
    train_data = data.DataLoader(config.train.data.train)
    valid_data = data.DataLoader(config.train.data.train | DictTree(num_unannotated=1000))
    test_data = data.DataLoader(config.train.data.test)
    saver = torch_utils.Saver(agent, train_data=train_data, valid_data=valid_data, test_data=test_data, config=DictTree(
        save_dir=config.train.logdir,
        keep_save_freq=config.train.keep_save_freq,
    ))
    step, restored_step = saver.restore()
    best_valid_loss = math.inf
    best_step = None
    last_save_step = restored_step
    last_eval_step = -math.inf
    stats_writer = tf_utils.TensorBoardWriter(config.train.logdir)

    try:
        while True:
            for train_batch in train_data.batches(step):
                if step >= config.train.num_steps:
                    return
                if config.train.save_freq and step - last_save_step >= config.train.save_freq and step > restored_step:
                    saver.save(step)
                    last_save_step = step
                if config.train.eval_freq and step - last_eval_step >= config.train.eval_freq:
                    valid_loss = _evaluate(valid_data, agent, 'valid', stats_writer, step)
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_step = step
                    last_eval_step = step
                train_stats = _step(agent, train_batch)
                step += len(train_batch)
                # TODO: more stats
                # TODO: summarize stats for entire training epoch?
                avg_stats = train_stats.get('per_trace', DictTree()) / len(train_batch)
                avg_stats |= train_stats.get('per_step', DictTree()) / sum(
                    trace.metadata.length for trace in train_batch)
                for k, v in avg_stats.allitems():
                    k = '/'.join(k)
                    # print(f"Step {step} training {k}: {v}")
                    stats_writer.add(step, f'train/{k}', v)
                stats_writer.flush()
    finally:
        try:
            if step > restored_step:
                saver.save(step)
            saver.restore(step=best_step)
            _evaluate(test_data, agent, 'test', stats_writer, step)
        finally:
            stats_writer.close()


def _evaluate(test_data, agent, test_str, stats_writer, step):
    num_traces = 0
    num_steps = 0
    test_stats = DictTree()
    for test_batch in test_data.batches():
        num_traces += len(test_batch)
        num_steps += sum(trace.metadata.length for trace in test_batch)
        stats = _step(agent, test_batch, train_mode=False)
        stats |= agent.evaluate(test_batch)
        test_stats += stats
    avg_stats = test_stats.get('per_trace', DictTree()) / num_traces
    avg_stats |= test_stats.get('per_step', DictTree()) / num_steps
    timestamp = datetime.now().replace(microsecond=0).isoformat()
    for k, v in avg_stats.allitems():
        k = '/'.join(k)
        print(f"[{timestamp}] Step {step} {test_str} {k}: {v}")
        stats_writer.add(step, f'{test_str}/{k}', v)
    stats_writer.flush()
    if 'loss' in avg_stats:
        return avg_stats.loss
    else:
        return avg_stats.score


def _step(agent, batch, train_mode=True):
    agent.train(train_mode)  # set to evaluation or training mode
    with contextlib.ExitStack() as stack:
        if not train_mode:
            stack.enter_context(torch.no_grad())
        stats = agent.get_loss(batch)
        if train_mode:
            agent.opt_step(stats.loss)
    agent.train()  # set to training mode
    del stats.loss
    return stats


def main():
    config_keys = ['env', 'agent', 'train']
    parser = argparse.ArgumentParser()
    for k in config_keys:
        parser.add_argument(f'--{k}', required=True)
    parser.add_argument('--cont')
    args = parser.parse_args()
    config = utils.json2dict(args, config_keys)

    config.train = DEFAULT_CONFIG | config.train
    config.train.trial_datetime = args.cont or config.train.trial_datetime or datetime.now().strftime(
        '%Y-%m-%d %H-%M-%S')

    train(config)


if __name__ == '__main__':
    main()
