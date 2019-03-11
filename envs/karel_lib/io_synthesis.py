# main parse file: karel-pytorch/data/karel/check_with_interp.py

import argparse
import collections
import hashlib
import itertools
import json
import pickle
import struct
import random
import time

import numpy as np
import tqdm

from os import system
from os.path import dirname, abspath, join

from .karel_runtime import KarelRuntime
from .utils import Timeout, TimeoutError
from .parser_for_synthesis import KarelForSynthesisParser
from .coverage import CoverageMeasurer

from .semantic_synthesizers import UniformBernoulliModel

IOSynthesisParameters = collections.namedtuple('IOSynthesisParameters',
                                               [
                                                    'max_timeout',
                                                    'max_runtime',
                                                    'max_bad_cover',
                                                    'wall_position_generator',
                                                    'marker_position_generator',
                                                    'covers_optional',
                                                    'constant_params_given_program',
                                                    'num_per_group',
                                                    'synthetic_held_out'
                                               ])

def get_marker_param(distributions):
    return distributions['seed'].rand()

class CouldNotGenerateConsistentIO(Exception):
    pass

class InvalidIOException(Exception):
    pass

class CrashingIOException(InvalidIOException):
    pass

class TimeoutIOException(InvalidIOException):
    pass

class InputGenerator:
    def __init__(self, distributions, io_synthesis_params):
        self.distributions = distributions
        self.io_synthesis_params = io_synthesis_params
        self.master_karel = KarelRuntime()

class RandomInputGridGenerator(InputGenerator):
    def generate_input_list(self, rng, p):
        if p is None:
            p = get_marker_param(self.distributions)
        ws = self.distributions['world_size']()
        mm = self.distributions['max_marker_in_cell']()
        mc = self.distributions['marker_counts'](ws)
        mpg = UniformBernoulliModel(lambda: p, "Bernoulli(" + str(p) + ")")
        self.master_karel.init_randomly(
                ws, mm, mc, rng,
                wall_position_generator=self.io_synthesis_params.wall_position_generator,
                marker_position_generator=self.io_synthesis_params.marker_position_generator
        )
        return self.master_karel.world


class IOSynthesizer:
    parser = KarelForSynthesisParser(build_tree=False)

    def __init__(self,
                 program,
                 prog_idx,
                 distributions,
                 io_synthesis_params,
                 input_generator_type,
                 val_data=None):
        self.program = tuple(program)
        self.prog_idx = prog_idx
        self.parsed_program = self.parser.parse(program, debug=False)
        self.distributions = distributions
        self.io_synthesis_params = io_synthesis_params
        self.crashes = self.timeouts = self.io_generated = 0
        self.bad_cover = self.covering_io_generated = 0
        self.cov = CoverageMeasurer(program)
        self.everything = self.cov.uncovered_set()
        self.val_data = val_data
        self.random_pair_generator = input_generator_type(self.distributions, self.io_synthesis_params)
        if self.val_data and not self.io_synthesis_params.synthetic_held_out:
            assert self.val_data[self.prog_idx][0] == self.program, \
                'val_data[prog_idx] not for specified program: {} vs {}'.format(
                            self.val_data[self.prog_idx],
                            self.program)
        else:
            assert self.io_synthesis_params.synthetic_held_out, \
                "Need to provide validation data if the held out data " \
                "isn't synthetic"

    def stats(self):
        return self.crashes, self.timeouts, self.io_generated, self.bad_cover, self.covering_io_generated

    def check_limits(self):
        if self.timeouts >= self.io_synthesis_params.max_timeout \
                or self.crashes >= self.io_synthesis_params.max_runtime \
                or self.bad_cover >= self.io_synthesis_params.max_bad_cover:
            raise CouldNotGenerateConsistentIO

    def io_crashed(self):
        self.crashes += 1
        self.check_limits()
    def io_timedout(self):
        self.timeouts += 1
        self.check_limits()
    def io_not_cover(self):
        self.bad_cover += 1
        self.check_limits()

    def get_subset_dicts(self, num_io):
        """return a list of num_io (covering) I/O sets for a program,
        where each set contains `self.io_synthesis_params.num_per_group` + 1 I/O
        pairs.
        """
        io_sets = []
        for _ in range(num_io):
            io_sets.append(self.generate_covering_io())
        if self.io_synthesis_params.synthetic_held_out:
            val_ios = [s[-1] for s in io_sets]
            io_sets = [s[:self.io_synthesis_params.num_per_group] for s in io_sets]
        else:
            val_code, val_io = self.val_data[self.prog_idx]
            val_ios = [val_io] * len(io_sets)

        data_dicts = []
        for io_subset, given_val_io in zip(io_sets, val_ios):
            data_dicts.append(
                    to_data_dict(self.program, io_subset, given_val_io))
        return data_dicts


    def generate_covering_io(self):
        """Return a list of io pairs that fully cover the program
        (unless covers_optional)"""
        rng = np.random.RandomState(self.distributions['seed'].randint(1, 2 ** 32))
        if self.io_synthesis_params.constant_params_given_program:
            p = get_marker_param(self.distributions)
        else:
            p = None
        while True:
            examples = [
                self.generate_consistent_io_example(rng, p)
                for _ in range(self.io_synthesis_params.num_per_group)
            ]
            if not self.io_synthesis_params.covers_optional:
                coverage = set()
                for eg in examples[:self.io_synthesis_params.num_per_group]:
                    self.cov.add(eg['in'])
                    coverage |=  self.cov.covered_set()
                    self.cov.reset()
            if (self.io_synthesis_params.covers_optional or
                    coverage == self.everything):
                self.covering_io_generated += 1
                break
            self.io_not_cover()

        if self.io_synthesis_params.synthetic_held_out:
            examples.append(self.generate_consistent_io_example(rng, p))

        return examples


    def generate_consistent_io_example(self, rng, p):
        """Return an IO pair consistent with the program"""
        while True:
            try:
                in_list, out_list = self.attempt_to_generate_consistent_io_example(rng, p)
                self.io_generated += 1
                return {'in': in_list, 'out': out_list}
            except CrashingIOException:
                self.io_crashed()
            except TimeoutIOException:
                self.io_timedout()

    def evaluate_on_grid(self, input_state):
        timeout = Timeout(1000)
        def action_callback(name, success, metadata):
            del name, metadata
            timeout.inc()
            if not success:
                raise CrashingIOException

        def event_callback(*_):
            timeout.inc()

        self.parser.karel.init_from_array(input_state)
        self.parser.karel.action_callback = action_callback
        self.parser.karel.event_callback = event_callback
        # need to check updated repo KarelRuntime event_callback
        try:
            self.parsed_program()
        except TimeoutError:
            raise TimeoutIOException

    def attempt_to_generate_consistent_io_example(self, rng, p):
        """Try an input grid and see if consistent. Repeat until end condition"""
        input_state = self.random_pair_generator.generate_input_list(rng, p)
        in_list = np.nonzero(input_state.ravel())[0].astype("uint16")
        self.evaluate_on_grid(input_state)
        output_state = input_state # mutated by parsed
        out_list = np.nonzero(output_state.ravel())[0].astype("uint16")
        return in_list, out_list


def batch_iter(iterable, size):
    sourceiter = iter(iterable)
    while True:
        batchiter = itertools.islice(sourceiter, size)
        yield itertools.chain([next(batchiter)], batchiter)


def to_data_dict(code, io_subsets, val_io):
    """Format data to be pickled"""
    return {
        'id': None,
        'guid': None,
        'code': code,
        'examples': io_subsets + [val_io]
    }


def run_evaluator(dataset_name, model, batch_create_eval="ConstantBatch(5, 1, False)"):
    """Evaluate model on a dataset"""
    exec_command = "cd ../program_synthesis; python eval.py --model_type=karel-lgrl --dataset karel" \
                + " --max_beam_trees 16 --max_eval_trials 1" \
                + " --model_dir {model}" \
                + " --batch-create-eval '{batch_create_eval}'" \
                + " --report-path {report}" \
                + " --eval-data-path {data}" \
                + " --hide-example-info"
    exec_command = exec_command.format(
        model = model if model[0] == "/" else "logdirs/{model}/".format(model=model),
        report = 'reports/report-%s-%s-%s.json' % (batch_create_eval.replace(" ", "_").replace(")", "_").replace("(", "_"), model.replace("/", "-"), dataset_name.replace("/", "-")),
        data = dataset_name if dataset_name[0] == "/" else 'analysis/synthesized/%s.pkl' % dataset_name,
        batch_create_eval=batch_create_eval)

    print(exec_command)
    system(exec_command)


def synthesize_io_for_single_program(program, prog_idx, distributions,
        io_synthesis_params, input_generator_type, num_io):
    synth = IOSynthesizer(
        program,
        prog_idx,
        distributions,
        io_synthesis_params,
        input_generator_type,
        val_data=UNIQUE_VAL_DATA)
    data_dicts, stats = [], []

    try:
        for data_dict in synth.get_subset_dicts(num_io=num_io):
            m = hashlib.sha1()
            m.update(' '.join(data_dict['code']).encode('utf-8'))
            for example in data_dict['examples']:
                if not isinstance(example['in'], np.ndarray):
                    example['in'] = np.array(example['in'], dtype=np.uint16)
                if not isinstance(example['out'], np.ndarray):
                    example['out'] = np.array(example['out'], dtype=np.uint16)
                m.update(example['in'])
                m.update(example['out'])
            data_dict['guid'] = m.hexdigest()[:16]

            data_dicts.append(data_dict)
            stats.append(synth.stats())
    except CouldNotGenerateConsistentIO:
        return None, None

    return data_dicts, stats

