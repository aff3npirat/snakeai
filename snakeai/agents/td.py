import random
from collections import defaultdict

from snakeai.helper import write_to_file, dict_to_str, default_value


# TODO: TDQLearning
class TDSarsa:

    def __init__(self, params, name, view, eps_greedy):
        params["n_games"] = 0
        self.params = params
        self.Q = defaultdict(default_value)
        self.name = name
        self.view = view
        self.eps_greedy = eps_greedy

    def get_action(self, state):
        action_probs = self.eps_greedy(self.Q[state], self.params)
        return random.choices([0, 1, 2, 3], weights=action_probs)[0]

    def get_state(self, game):
        return self.view(game)

    def train_episode(self, game, render):
        game.reset()
        done = False
        state = self.get_state(game)
        action = self.get_action(state)
        while not done:
            done, reward = game.play_step(action, render)
            next_state = self.get_state(game)
            next_action = self.get_action(next_state)
            target = reward + self.params['gamma'] * self.Q[next_state][next_action]
            delta = target - self.Q[state][action]
            self.Q[state][action] += self.params['lr'] * delta
            state = next_state
            action = next_action
        self.params['n_games'] += 1

    def save(self, save_dir):
        write_to_file(self, save_dir / f"{self.name}.pkl", text=False)
        info = (f"{type(self).__name__}\n"
                f"{dict_to_str(self.params)}")
        write_to_file(info, save_dir / f"{self.name}.yml", text=True)
        print(f"Saved {self.name} to '{save_dir}'")
