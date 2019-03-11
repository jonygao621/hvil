import argparse
import functools
import cPickle as pickle

import numpy as np
import ray
import tqdm

from analysis import io_synthesis
from analysis import semantic_synthesizers
from program_synthesis.datasets import indexed_file

distributions = {
    'seed': np.random,
    'world_size': functools.partial(np.random.randint, 2, 17, 2),
    'max_marker_in_cell': functools.partial(np.random.randint, 1, 10),
    'marker_counts': semantic_synthesizers.MarkerCountGenerator(
        semantic_synthesizers.uniform_markers, [1, 9]),
    'hard_params': False
}


@ray.remote
def replace_held_out(entry, count):
    io_synthesis_params = io_synthesis.IOSynthesisParameters(
        max_timeout=1e8,
        max_runtime=1e8,
        max_bad_cover=1e8,
        wall_position_generator=semantic_synthesizers.UniformBernoulliModel(
            np.random.rand, 'uniform (0, 1)'),
        marker_position_generator=semantic_synthesizers.UniformBernoulliModel(
            np.random.rand, 'uniform (0, 1)'),
        covers_optional=True,
        constant_params_given_program=False,
        num_per_group=count,
        synthetic_held_out=True)

    synth = io_synthesis.IOSynthesizer(
        entry['code'],
        prog_idx=None,
        distributions=distributions,
        io_synthesis_params=io_synthesis_params,
        input_generator_type=io_synthesis.RandomInputGridGenerator)
    entry['examples'] += synth.generate_covering_io()[:-1]
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-shown', type=int, default=5)
    parser.add_argument('--num-held-out', type=int, required=True)
    args = parser.parse_args()

    ray.init()

    f = open(args.input)
    index = indexed_file.read_index(args.input + '.index')
    out = indexed_file.IndexedFileWriter(args.output)
    pbar = tqdm.tqdm(total=len(index), dynamic_ncols=True)

    results = []
    while True:
        try:
            entry = pickle.load(f)
            del entry['examples'][args.num_shown:]
            results.append(replace_held_out.remote(entry, args.num_held_out))
        except EOFError:
            break

    while results:
        ready, results = ray.wait(results)
        for r in ready:
            out.append(pickle.dumps(ray.get(r), pickle.HIGHEST_PROTOCOL))
            pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()
