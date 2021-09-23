import random
from datetime import datetime

from snakeai import root_dir as root_dir_
from snakeai.helper import plot, save_plot, read_from_file, write_to_file, dict_to_str
from snakeai.snake_game import SnakeGame


# TODO: implement q-learning, which runs better?
class TDSarsa:

    def __init__(self, params, name):
        self.params = params
        self.Q = {}
        self.name = name

    def train_episode(self, game, get_state, get_action, render):
        game.reset()
        done = False
        state = get_state(game)
        action = get_action(state)
        while not done:
            reward, done = game.play_step(action, render)
            next_state = get_state(game)

            if state not in self.Q:
                self.Q[state] = [0, 0, 0, 0]
            if next_state not in self.Q:
                self.Q[next_state] = [0, 0, 0, 0]

            next_action = get_action(state)
            target = reward + self.params['gamma'] * self.Q[next_state][next_action]
            delta = target - self.Q[state][action]
            self.Q[state][action] += self.params['lr'] * delta
        self.params['n_games'] += 1

    def save(self, root_dir):
        write_to_file(self, root_dir / f"{self.name}.pkl", text=False)
        write_to_file(dict_to_str(self.params), root_dir / f"{self.name}.yml", text=True)
        print(f"Saved agent to '{root_dir}'")
