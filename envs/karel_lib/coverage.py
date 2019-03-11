import collections
import itertools

from . import executor
from . import parser_for_synthesis

branch_types = {'if', 'ifElse', 'while'}
stmt_types = {'move', 'turnLeft', 'turnRight', 'putMarker', 'pickMarker'}


class CoverageMeasurer(object):
    parser = parser_for_synthesis.KarelForSynthesisParser(build_tree=True)
    executor = executor.KarelExecutor()

    def __init__(self, code):
        self.code = code
        self.tree = self.parser.parse(code)

        self.action_spans = []
        self.cond_block_spans = []

        queue = collections.deque(self.tree['body'])
        while queue:
            node = queue.popleft()
            if node['type'] in stmt_types:
                self.action_spans.append(node['span'])
            elif node['type'] in branch_types:
                self.cond_block_spans.append(node['span'])
            queue.extend(node.get('body', []))
            queue.extend(node.get('elseBody', []))
            queue.extend(node.get('ifBody', []))

        self.reset()

    def add(self, inp):
        out, trace = self.executor.execute(
            self.code, None, inp, record_trace=True)
        if not out:
            return False

        for event in trace.events:
            if event.type in branch_types:
                self.branch_coverage[event.span, event.cond_value] += 1
            elif event.type in stmt_types:
                self.stmt_coverage[event.span] += 1

        return True

    def reset(self):
        # Statement coverage: actions
        self.stmt_coverage = {span: 0 for span in self.action_spans}
        # Branch coverage: if, ifelse, while
        self.branch_coverage = {(span, cond_value): 0
                                for span in self.cond_block_spans
                                for cond_value in (True, False)}

    def uncovered(self):
        return (tuple(k for k, v in self.stmt_coverage.items() if v == 0),
                tuple(k for k, v in self.branch_coverage.items() if v == 0))

    def uncovered_set(self):
        return set(k for k, v in itertools.chain(self.stmt_coverage.items(),
                                                 self.branch_coverage.items())
                   if v == 0)

    def covered_set(self):
        return set(k for k, v in itertools.chain(self.stmt_coverage.items(),
                                                 self.branch_coverage.items())
                   if v != 0)


def uncovered_after_merge(measurers):
    """
    Returns the uncovered code after merging the code coverage
    from all of the CoverageMeasurers in measurers (a list).
    """

    def get_intersection(tuples):
        sets = [set(t) for t in tuples]
        return sorted(tuple(reduce(lambda x, y: x.intersection(y), sets)))

    unc_stmts = get_intersection([m.uncovered()[0] for m in measurers])
    unc_branches = get_intersection([m.uncovered()[1] for m in measurers])

    return (tuple(unc_stmts), tuple(unc_branches))
