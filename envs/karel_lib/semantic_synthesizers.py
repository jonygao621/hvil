
import numpy as np
from os import system
from sys import argv
#from scipy.misc import imsave, imresize

class TupleHashed(object):
    def __hash__(self):
        return hash(self.to_tuple())
    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()
    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, ", ".join(str(x) for x in self.to_tuple()))
    def to_tuple(self):
        raise RuntimeError

class LineSegmentModel(TupleHashed):
    def __init__(self, perc_lines, avg_len_frac, std_len_frac):
        self.perc_lines = perc_lines
        self.avg_len_frac = avg_len_frac
        self.std_len_frac = std_len_frac
    def __call__(self, rng, width, height, prev):
        result = np.zeros((width, height), dtype=np.int8)
        for x in rng.choice(width, size=int(np.round(width * self.perc_lines))):
            result[x, self.create_line(rng, height)] = 1
        for y in rng.choice(height, size=int(np.round(height * self.perc_lines))):
            result[self.create_line(rng, width), y] = 1
        return result & ~prev
    def create_line(self, rng, length):
        seg_length = int(np.round((rng.randn() * self.std_len_frac + self.avg_len_frac) * length))
        if seg_length < 0:
            seg_length = 0
        if seg_length >= length:
            seg_length = length - 1
        start_pos = rng.choice(length - seg_length)
        return slice(start_pos, start_pos + seg_length)
    def to_tuple(self):
        return (self.perc_lines, self.avg_len_frac, self.std_len_frac)

class CurveModel(TupleHashed):
    def __init__(self, perc_length, prob_turn, perc_keep):
        self.perc_length = perc_length
        self.prob_turn = prob_turn
        self.perc_keep = perc_keep
    def __call__(self, rng, width, height, prev):
        result = np.zeros((width, height), dtype=np.int8)
        x, y = rng.choice(width), rng.choice(height)
        direction = 0
        DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for _ in range(int(width * height * self.perc_length)):
            if rng.rand() < self.perc_keep:
                result[x, y] = 1
            if rng.rand() < self.prob_turn:
                direction += rng.choice([-1, 1])
                direction %= 4
            dx, dy = DIRECTIONS[direction]
            x += dx
            y += dy
            x %= width
            y %= height
        return result & ~prev
    def to_tuple(self):
        return (self.perc_length, self.prob_turn, self.perc_keep)

class UniformBernoulliModel(TupleHashed):
    def __init__(self, prob_select_generator, repr_string):
         self.prob_select_generator = prob_select_generator
         self.latest_prob = None
         self.repr_string = repr_string
    def __call__(self, rng, height, width, prev):
        self.latest_prob = self.prob_select_generator()
        return (rng.rand(height, width) < self.latest_prob) & ~prev
    def to_tuple(self):
        return (self.latest_prob,)
    def __repr__(self):
        return self.repr_string

class UniformSubsetModel(TupleHashed):
    def __init__(self, prob_select_generator, repr_string):
        self.prob_select_generator = prob_select_generator
        self.latest_prob = None
        self.repr_string = repr_string
    def __call__(self, rng, height, width, prev):
        locs = [(y, x) for y  in range(height) for x in range(width) if not prev[y][x]]
        result = np.zeros((height, width)).astype(int)
        self.latest_prob = self.prob_select_generator()
        assert len(locs) >= int(height * width * self.latest_prob)

        for idx in rng.choice(len(locs), size=int(height * width * self.latest_prob), replace=False):
            result[locs[idx]] = 1

        try:
            assert int(height * width * self.latest_prob) == np.sum(result)
        except AssertionError:
            print(int(height*width*self.latest_prob), np.sum(result))
        return result
    def to_tuple(self):
        return (self.latest_prob,)
    def __repr__(self):
        return self.repr_string


class MarkerCountGenerator:
    def __init__(self, function, function_args):
        self.function = function(*function_args)

    def __call__(self, shape):
        return self.function(shape)

# Different distribution options
def uniform_markers(low, high):
    """Return a uniform number of markers between low and high (inclusive)."""
    low, high = int(low), int(high)
    return lambda shape: np.random.randint(low, high + 1, size=shape)

def geometric_markers(p):
    """Geometric distribution of markers. lower p -> steeper dropoff."""
    p = float(p)
    return lambda shape: np.random.geometric(p, size=shape)

def anti_geometric_markers(p):
    """Geometric distribution but backwards. lower p -> steeper dropoff from 9."""
    def generator(shape):
        geom = geometric_markers(p)(shape)
        geom[geom > 9] = 9
        return 10 - geom

    return generator

def bimodal_normal(low, high, stddev):
    """Two normal distributions in one."""
    def for_shape(shape):
        mask = np.random.rand(size=shape) < 0.5
        low_distro = np.round(np.random.normal(low, stddev, size=shape))
        high_distro = np.round(np.random.normal(high, stddev, size=shape))
        sample = mask * low_distro + (~mask) * high_distro
        return np.clip(sample, 1, 9).astype(int)
    return for_shape

def exact_pdf(pdf):
    """Design your own. pdf need not be normalized to begin with."""
    normed_pdf = pdf / np.linalg.norm(pdf, 1)
    return lambda shape: np.random.choice(range(1, len(pdf)), p=normed_pdf, size=shape)

if __name__ == '__main__':
    _, small_size, large_size, n_render, filepath, model = argv
    render_several((int(small_size), int(large_size)), int(n_render), filepath, eval(model))
