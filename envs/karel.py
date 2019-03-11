import collections

import numpy as np
import torch
import tqdm

#from php import agent
#from php import modules
from utils import DictTree
from . import env
from .karel_lib import io_synthesis
from .karel_lib import karel_runtime
from .karel_lib import semantic_synthesizers
from .karel_lib import parser_for_synthesis


def _observe(kr):
    obs = [
        kr.frontIsClear(),
        kr.leftIsClear(),
        kr.rightIsClear(),
        kr.markersPresent()
    ]
    return torch.tensor([int(x) for x in obs], dtype=torch.float)

DEFAULT_CONFIG = DictTree(
        num_grids=1100,
        max_grid_size=5,
        program='cs106a:0',
)


class Karel(env.Environment):
    def __init__(self, config):
        self.config = DEFAULT_CONFIG | config
        super().__init__()

        self.parser = parser_for_synthesis.KarelForSynthesisParser()
        self.kr = self.parser.karel
        self.grids_id = -1 
        self.full_grid = np.empty((15, 18, 18), dtype=np.bool)
        self.actions_list = None

        # Generate environment ahead of time
        self.program = ALL_PROGRAMS[self.config.program].split()
        self.code = self.parser.parse(self.program)
        self.all_io_grids = generate_io(self.program, self.config.num_grids,
                    self.config.max_grid_size)

    class move(env.Action):
        arg_in_size = 0
        ret_out_size = 0
        async def __call__(self, iput):
            iput.env.kr.move()
            return torch.empty(0)
    class turnLeft(env.Action):
        arg_in_size = 0
        ret_out_size = 0
        async def __call__(self, iput):
            iput.env.kr.turnLeft()
            return torch.empty(0)
    class turnRight(env.Action):
        arg_in_size = 0
        ret_out_size = 0
        async def __call__(self, iput):
            iput.env.kr.turnRight()
            return torch.empty(0)
    class pickMarker(env.Action):
        arg_in_size = 0
        ret_out_size = 0
        async def __call__(self, iput):
            iput.env.kr.pickMarker()
            return torch.empty(0)
    class putMarker(env.Action):
        arg_in_size = 0
        ret_out_size = 0
        async def __call__(self, iput):
            iput.env.kr.putMarker()
            return torch.empty(0)
    actions = {action.__name__: action() for action in [move, turnLeft,
        turnRight, pickMarker, putMarker]}

    # init_arg_size

    @property
    def term_ret_size(self):
        return 0

    @property
    def obs_size(self):
        return 4

    def reset(self):
        self.grids_id += 1
        if self.grids_id >= len(self.all_io_grids):
            raise ValueError('Pre-generated grids exhausted')
        current_grid = self.all_io_grids[self.grids_id]['in']

        self.full_grid[:] = 0
        self.full_grid.ravel()[current_grid] = 1
        self.kr.init_from_array(self.full_grid)
        actions = []
        def cb(action, _):
            actions.append(action)
        old_cb = self.kr.pre_action_callback
        self.kr.pre_action_callback = cb
        self.code()
        self.kr.pre_action_callback = old_cb
        actions.append(None)
        self.actions_list = actions

        self.full_grid[:] = 0
        self.full_grid.ravel()[current_grid] = 1
        self.kr.init_from_array(self.full_grid)

        return DictTree(value=torch.empty(0))

    def observe(self):
        return DictTree(value=_observe(self.kr))

    def print_state(self):
        self.kr.draw()


def generate_io(program, count, max_world_size=5):
    dists = {
        'seed':
        np.random.RandomState(734),
        'world_size':
        lambda: dists['seed'].randint(2, max_world_size, 2),
        'max_marker_in_cell':
        lambda: 9,
        'marker_counts':
        semantic_synthesizers.MarkerCountGenerator(
            semantic_synthesizers.geometric_markers, [0.6]),
    }
    seed = dists['seed']

    def simulated_marker_ratio():
        marker_rand = seed.rand()
        if marker_rand < 0.25:
            return seed.uniform(0, 0.04)
        elif marker_rand > 0.96:
            return seed.uniform(0.15, 0.6)
        else:
            return max(0, seed.normal(0.06, 0.025))

    def simulated_wall_ratio():
        wall_rand = seed.rand()
        if wall_rand < 0.32:
            return seed.uniform(0, 0.04)
        elif wall_rand > 0.96:
            return seed.uniform(0.15, 0.4)
        else:
            return max(0, seed.normal(0.06, 0.025))

    params = io_synthesis.IOSynthesisParameters(
        max_timeout=1e8,
        max_runtime=1e8,
        max_bad_cover=1e8,
        wall_position_generator=semantic_synthesizers.UniformBernoulliModel(
            simulated_wall_ratio, 'simulated_wall_ratio_generator'),
        marker_position_generator=semantic_synthesizers.UniformBernoulliModel(
            simulated_marker_ratio, 'simulated_marker_ratio_generator'),
        covers_optional=False,
        constant_params_given_program=False,
        num_per_group=5,
        synthetic_held_out=True)
    synth = io_synthesis.IOSynthesizer(
        program,
        prog_idx=None,
        distributions=dists,
        io_synthesis_params=params,
        input_generator_type=io_synthesis.RandomInputGridGenerator)

    result = []
    for _ in tqdm.trange((count + 4) // 5, desc='generating grids'):
        result += synth.generate_covering_io()[:-1]
    return result



def create_expert_trajectories(env_args, count):
    parser = parser_for_synthesis.KarelForSynthesisParser()
    env_args = eval(env_args)
    program = ALL_PROGRAMS[env_args['program']].split()

    primitives = {
        name: modules.Primitive(name)
        for name in ('move', 'turnLeft', 'turnRight', 'pickMarker',
                     'putMarker')
    }

    max_grid_size = env_args.get('max_grid_size', 5)
    all_io_grids = generate_io(program, count, max_grid_size)

    code = parser.parse(program)
    trajs = []
    full_grid = np.empty((15, 18, 18), dtype=np.bool)
    for io in tqdm.tqdm(all_io_grids):
        observations = []
        stacks = []
        actions = []
        action_args = []
        step_infos = []
        root_arg = 0

        stack = [agent.StackEntry(php='run', state=0, arg=0)]
        keep_parent_state = True

        def pre_action_callback(name, metadata):
            #print('action: {}, {}'.format(name, metadata))
            nonlocal keep_parent_state
            if not keep_parent_state:
                increment_top(stack)

            observations.append(_observe(parser.karel))
            stacks.append(
                tuple(
                    agent.StackEntry(
                        php=entry.php,
                        state=torch.tensor([entry.state], dtype=torch.float),
                        arg=torch.tensor(entry.arg)) for entry in
                    (stack +
                     [agent.StackEntry(
                         php=name + '_php',
                         state=0,
                         arg=0,
                     )])))
            actions.append(primitives[name])
            action_args.append(torch.tensor(0))
            step_infos.append(None)

            keep_parent_state = False

        def block_event_callback(name, span, cond_value, outcome):
            nonlocal keep_parent_state
            #print('block: {}, {}, {}, {}'.format(name, span, cond_value, outcome))
            if name == 'if' and cond_value:
                php_name = 'if{}_body'.format(span[0])
                if outcome == 'start':
                    if not keep_parent_state:
                        increment_top(stack)
                    stack.append(
                        agent.StackEntry(
                            php=php_name,
                            state=0,
                            arg=0,
                        ))
                elif outcome == 'end':
                    assert stack[-1].php == php_name
                    stack.pop()
                keep_parent_state = True

            elif name == 'ifElse':
                if cond_value:
                    php_name = 'ifElse{}_if_body'.format(span[0])
                else:
                    php_name = 'ifElse{}_else_body'.format(span[0])

                if outcome == 'start':
                    if not keep_parent_state:
                        increment_top(stack)
                    stack.append(
                        agent.StackEntry(
                            php=php_name,
                            state=0,
                            arg=0,
                        ))
                elif outcome == 'end':
                    assert stack[-1].php == php_name
                    stack.pop()
                keep_parent_state = True

            elif name == 'while':
                php_name = 'while{}_body'.format(span[0])
                wrapper_name = 'while{}_wrapper'.format(span[0])
                # If loop never runs, then body should be never invoked
                if outcome == 'enter' and cond_value:
                    if stack[-1].php == php_name:
                        # Loop is already running
                        # 1. Remove while_body from stack so that it can be
                        # re-added with a state of 0
                        stack.pop()
                        # 2. Increment tau of while_wrapper
                        assert stack[-1].php == wrapper_name
                        increment_top(stack)
                    else:
                        # Loop is not running
                        if not keep_parent_state:
                            increment_top(stack)
                        # Put while_wrapper
                        stack.append(
                            agent.StackEntry(
                                php=wrapper_name,
                                state=0,
                                arg=0,
                            ), )

                    stack.append(
                        agent.StackEntry(
                            php=php_name,
                            state=0,
                            arg=0,
                        ), )
                    keep_parent_state = True
                elif outcome == 'exit':
                    if stack[-1].php == php_name:
                        # Remove while_body
                        stack.pop()
                        # Remove while_wrapper
                        assert stack[-1].php == wrapper_name
                        stack.pop()
                        #keep_parent_state = True
                        keep_parent_state = False

            elif name == 'repeat':
                php_name = 'repeat{}_body'.format(span[0])
                if outcome == 'enter':
                    if stack[-1].php == php_name:
                        stack.pop()

                    if not keep_parent_state:
                        increment_top(stack)

                    stack.append(
                        agent.StackEntry(
                            php=php_name,
                            state=0,
                            arg=0,
                        ))
                    keep_parent_state = True

                elif outcome == 'exit':
                    assert stack[-1].php == php_name
                    stack.pop()
                    keep_parent_state = True

            #elif name == 'while':
            #    php_name = 'while{}_body'.format(span[0])
            #    # If loop never runs, then body should be never invoked
            #    if outcome == 'enter' and cond_value:
            #        if stack[-1].php == php_name:
            #            # Loop is already running
            #            stack.pop()
            #        stack.append(
            #            agent.StackEntry(php=php_name, state=0, arg=0, reps=1))
            #        keep_parent_state = True
            #    elif outcome == 'exit':
            #        if stack[-1].php == php_name:
            #            stack.pop()

            #elif name == 'repeat':
            #    php_name = 'repeat{}_body'.format(span[0])
            #    if outcome == 'enter':
            #        if stack[-1].php == php_name:
            #            assert stack[-1].reps == cond_value + 1
            #            stack.pop()

            #        stack.append(
            #            agent.StackEntry(
            #              php=php_name, state=0, arg=0, reps=cond_value))
            #        keep_parent_state = True

            #    elif outcome == 'exit':
            #        assert stack[-1].php == php_name
            #        assert stack[-1].reps == 1
            #        stack.pop()

        full_grid[:] = 0
        full_grid.ravel()[io['in']] = 1
        parser.karel.init_from_array(full_grid)
        parser.karel.pre_action_callback = pre_action_callback
        parser.karel.block_event_callback = block_event_callback
        code()

        observations.append(_observe(parser.karel))
        stacks.append(())
        actions.append(None)
        action_args.append(None)
        step_infos.append(None)

        traj = agent.Trajectory(root_arg, observations, stacks, actions,
                                action_args, step_infos, {})
        trajs.append(traj)

    return trajs

def extract_metadata(traj):
    return ()

ALL_PROGRAMS = {
    'cs106a:0':
    'DEF run m( REPEAT R=4 r( putMarker move turnLeft r) m)',
    'cs106a:1':
    'DEF run m( REPEAT R=4 r( WHILE c( frontIsClear c) w( move putMarker w) turnLeft r) m)',
    'cs106a:2':
    'DEF run m( move WHILE c( markersPresent c) w( pickMarker w) move m)',
    'cs106a:3':
    'DEF run m( WHILE c( frontIsClear c) w( putMarker turnLeft IF c( frontIsClear c) i( move turnRight IF c( frontIsClear c) i( move i) i) w) m)',
    'cs106a:3a':
    '''DEF run m( WHILE c( frontIsClear c) w( putMarker turnLeft IF c(
    frontIsClear c) i( move turnRight IF c( frontIsClear c) i( move i) i) w)
    putMarker m)''',
    'cs106a:3b':
    '''DEF run m( move WHILE c( frontIsClear c) w( putMarker turnLeft IF c(
    frontIsClear c) i( move turnRight IF c( frontIsClear c) i( move i) i) w)
    m)''',
    'cs106a:4':
    'DEF run m( move WHILE c( markersPresent c) w( pickMarker move putMarker putMarker turnLeft turnLeft move turnLeft turnLeft w) move move m)',
    'cs106a:4a':
    'DEF run m( move WHILE c( markersPresent c) w( pickMarker move putMarker putMarker turnLeft turnLeft move turnLeft turnLeft w) m)',
    'cs106a:5':
    'DEF run m( REPEAT R=4 r( putMarker putMarker move turnLeft r) m)',
    'cs106a:6':
    'DEF run m( putMarker WHILE c( frontIsClear c) w( move putMarker w) turnLeft turnLeft WHILE c( frontIsClear c) w( move w) turnLeft turnLeft WHILE c( leftIsClear c) w( turnLeft move turnRight putMarker WHILE c( frontIsClear c) w( move putMarker w) turnLeft turnLeft WHILE c( frontIsClear c) w( move w) turnLeft turnLeft w) m)',
    'cs106a:7':
    'DEF run m( IF c( rightIsClear c) i( turnRight move putMarker turnLeft turnLeft move turnRight i) WHILE c( frontIsClear c) w( move IF c( rightIsClear c) i( turnRight move putMarker turnLeft turnLeft move turnRight i) w) m)',
    'cs106a:8':
    'DEF run m( REPEAT R=4 r( putMarker move r) m)',
    'cs106a:9':
    'DEF run m( REPEAT R=4 r( putMarker move turnLeft r) move turnLeft move turnRight REPEAT R=4 r( putMarker move turnLeft r) m)',
    'cs106a:10':
    'DEF run m( turnLeft move turnRight move REPEAT R=4 r( WHILE c( frontIsClear c) w( putMarker move w) turnLeft turnLeft move pickMarker turnRight r) m)',
    'cs106a:11':
    'DEF run m( WHILE c( noMarkersPresent c) w( IFELSE c( rightIsClear c) i( turnRight move i) ELSE e( WHILE c( not c( frontIsClear c) c) w( turnLeft w) move e) w) m)',
    'cs106a:11a':
    'DEF run m( turnLeft WHILE c( noMarkersPresent c) w( IFELSE c( rightIsClear c) i( turnRight move i) ELSE e( WHILE c( not c( frontIsClear c) c) w( turnLeft w) e) w) m)',
    'cs106a:12':
    'DEF run m( WHILE c( not c( frontIsClear c) c) w( turnLeft move turnRight move w) putMarker WHILE c( frontIsClear c) w( move turnRight move turnLeft w) m)',
    'cs106a:13':
    'DEF run m( WHILE c( frontIsClear c) w( move w) turnRight move turnLeft move pickMarker turnLeft turnLeft WHILE c( frontIsClear c) w( move w) turnRight move m)',
    'cs106a:14':
    'DEF run m( move REPEAT R=9 r( putMarker r) move m)',
    'cs106a:15':
    'DEF run m( IF c( markersPresent c) i( turnLeft REPEAT R=4 r( putMarker move r) turnLeft turnLeft WHILE c( frontIsClear c) w( move w) turnLeft i) WHILE c( frontIsClear c) w( move IF c( markersPresent c) i( turnLeft REPEAT R=4 r( putMarker move r) turnLeft turnLeft WHILE c( frontIsClear c) w( move w) turnLeft i) w) m)',
}
