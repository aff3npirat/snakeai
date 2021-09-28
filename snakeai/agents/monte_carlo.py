import random
from collections import defaultdict

from snakeai.helper import default_value, dict_to_str, write_to_file


# TODO: EveryVisitMC
class FirstVisitMC:

    def __init__(self, params, name, view, eps_greedy):
        params["n_games"] = 0
        self.params = params
        self.name = name
        self.Q = defaultdict(default_value)
        self.num_visits = defaultdict(default_value)
        self.view = view
        self.eps_greedy = eps_greedy

    def get_action(self, state):
        action_probs = self.eps_greedy(self.Q[state], self.params)
        return random.choices([0, 1, 2, 3], weights=action_probs)

    def get_state(self, game):
        return self.view(game)

    def train_episode(self, game, render):
        game.reset()
        episode = []
        done = False
        while not done:
            state = self.get_state(game)
            action = self.get_action(state)
            done, reward = game.play_step(action, render)
            episode.append((state, action, reward))
        self.params['n_games'] += 1
        total_reward = 0
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]

            total_reward = self.params['gamma'] * total_reward + reward
            if (state, action) not in [(s, a) for s, a, _ in episode[0:i]]:
                self.num_visits[state][action] += 1
                self.Q[state][action] += ((total_reward - self.Q[state][action])
                                          / self.num_visits[state][action])

    def save(self, save_dir):
        write_to_file(self, save_dir / f"{self.name}.pkl", text=False)
        info = (f"{type(self).__name__}\n"
                f"{dict_to_str(self.params)}")
        write_to_file(info, save_dir / f"{self.name}.yml", text=True)
        print(f"Saved {self.name} to '{save_dir}'")
