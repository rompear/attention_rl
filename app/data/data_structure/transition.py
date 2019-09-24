from collections import namedtuple

class Transition:
    struct = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))