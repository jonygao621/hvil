import argparse
import os
import pickle

import numpy as np
import tqdm

import agents
import envs
import utils
from utils import DictTree

DEFAULT_CONFIG = DictTree(
    generate=True,
    load=False,
)


def get_num_traces(config):
    num_traces = config.get('num_traces')
    if num_traces is None:
        return config.num_annotated + config.num_unannotated
    else:
        return num_traces


class DataLoader(object):
    def __init__(self, config, env=None, agent=None):
        self.config = config
        num_traces = get_num_traces(self.config)
        if not num_traces:
            raise ValueError
        if self.config.generate:
            self._generate(env, agent, num_traces, self.config.dirname)
        if self.config.load:
            num_saved = len(os.listdir(self.config.dirname))
            num_annotated = self.config.get('num_annotated')
            if num_annotated is None:
                num_unannotated = self.config.num_unannotated
                num_annotated = num_traces - num_unannotated
            else:
                num_unannotated = self.config.get('num_unannotated', num_traces - num_annotated)
            assert 0 < max(num_annotated, num_unannotated) <= num_traces == num_annotated + num_unannotated <= num_saved
            self.trace_idxs = [[idx, True] for idx in np.random.choice(num_saved, num_traces, False)]
            for i in np.random.choice(num_traces, num_unannotated, False):
                self.trace_idxs[i][1] = False

    def state_dict(self):
        return {'trace_idxs': self.trace_idxs}

    def load_state_dict(self, state_dict):
        self.trace_idxs = state_dict['trace_idxs']

    @staticmethod
    def _generate(env, agent, num_traces, dirname):
        if os.path.exists(dirname):
            raise ValueError(f"path {dirname} already exists")
        else:
            os.makedirs(dirname)
        for trace_idx in tqdm.trange(num_traces, desc="generating traces"):
            trace = agent.rollout(env)
            filename = os.path.join(dirname, f'{trace_idx}.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(trace, f)

    def batches(self, init_step=0):
        unannotated_throttling_steps = self.config.unannotated_throttling_steps or 0
        if unannotated_throttling_steps:
            unannotated_frac = min(1., init_step / unannotated_throttling_steps)
        else:
            unannotated_frac = 1.

        def _batch_iter():
            batch_size = self.config.batch_size or len(self.trace_idxs)
            trace_idxs = [trace_idx for trace_idx in self.trace_idxs if trace_idx[1]]
            unannotated_trace_idxs = [trace_idx for trace_idx in self.trace_idxs if not trace_idx[1]]
            if unannotated_trace_idxs:
                selected_unannotated_trace_idxs = np.random.choice(
                    len(unannotated_trace_idxs), int(len(unannotated_trace_idxs) * unannotated_frac))
                trace_idxs += [unannotated_trace_idxs[selected] for selected in selected_unannotated_trace_idxs]
            trace_idxs = np.random.permutation(trace_idxs)
            num_batches = (len(trace_idxs) - 1) // batch_size + 1
            batches = [
                trace_idxs[i * len(trace_idxs) // num_batches:(i + 1) * len(trace_idxs) // num_batches]
                for i in range(num_batches)]
            for batch in batches:
                traces = []
                for trace_idx, annotated in batch:
                    filename = os.path.join(self.config.dirname, f'{trace_idx}.pkl')
                    with open(filename, 'rb') as f:
                        trace = pickle.load(f)
                    trace.metadata.annotated = annotated
                    if not annotated:
                        for step in trace.data.steps:
                            del step.mem_in
                            del step.mem_out
                    traces.append(trace)
                yield sorted(traces, key=lambda trc: -trc.metadata.length)

        return _batch_iter()


def main():
    config_keys = ['env', 'agent', 'data']
    parser = argparse.ArgumentParser()
    for k in config_keys:
        parser.add_argument(f'--{k}', required=True)
    args = parser.parse_args()
    config = utils.json2dict(args, config_keys)

    env = envs.catalog(config.env)
    agent = agents.catalog(env, DictTree(teacher=True) | config.agent)
    agent.to(config.data.get('device', 'cuda'))
    env.training = True
    DataLoader(DEFAULT_CONFIG | config.data.train, env, agent)
    env.training = False
    DataLoader(DEFAULT_CONFIG | config.data.test, env, agent)


if __name__ == '__main__':
    main()
