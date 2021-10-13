import random
from collections import defaultdict

from . import helpers


def default_value():
    return [0, 0, 0, 0]


class QAgentBase:

    def __init__(self, params, name, vision, eps_greedy):
        self.Q = defaultdict(default_value)
        self.params = params.copy()
        self.params['n_games'] = 0
        self.name = name
        self.vision = vision
        self.eps_greedy = eps_greedy

    def train_episode(self, game):
        raise NotImplementedError

    def get_state(self, game):
        return self.vision(game)

    def get_action(self, state):
        action_probs = self.eps_greedy(self.Q[state], self.params)
        return random.choices([0, 1, 2, 3], weights=action_probs)[0]

    def save(self, save_dir):
        helpers.write_to_file(self, save_dir / f"{self.name}.pkl", text=False)
        to_yml = (f"{type(self).__name__} "
                  f"({self.vision.__name__}/{self.eps_greedy.__name__})\n"
                  f"{helpers.dict_to_str(self.params)}")
        helpers.write_to_file(to_yml, save_dir / f"{self.name}.yml", text=True)
