import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model

from snakeai.helper import dict_to_str, write_to_file


class QNet:

    def __init__(self, hidden_size, out_size, lr, loss=keras.losses.MeanSquaredError()):
        self.model = keras.Sequential([
            layers.Dense(hidden_size, activation="relu", name="hidden"),
            layers.Dense(out_size, name="out")
        ])
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)

    def __getitem__(self, state):
        return self.model(np.array([state], dtype=int))[0]


# TODO: QNetSarsa
class QNetLearning:

    def __init__(self, params, name, view, eps_greedy):
        params["n_games"] = 0
        self.params = params
        self.Q = QNet(params['hidden_size'], 3, params['lr'])
        self.name = name
        self.qnet_file = None
        self.view = view
        self.eps_greedy = eps_greedy
        self.memory = deque(maxlen=100_000)

    def get_action(self, state):
        action_probs = self.eps_greedy(self.Q[state], self.params)
        return random.choices([0, 1, 2, 3], weights=action_probs)

    def get_state(self, game):
        return self.view(game)

    def train_episode(self, game, render):
        game.reset()
        done = False
        state = self.get_state(game)
        while not done:
            action = self.get_action(state)
            done, reward = game.play_step(action, render)
            next_state = self.get_state(game)
            # train on time step
            self._train([state], [action], [reward], [next_state], [done], 1)
            self.memory.append((state, action, reward, next_state, done))
            state = next_state
        # train on full memory
        self._train(*list(zip(*self.memory)), 1000)
        self.params['n_games'] += 1

    def _train(self, states, actions, rewards, next_states, dones, batch_size):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states,
                                                           dones):
            # output of qnet has shape (1, 4)
            target = tf.unstack(self.Q[state])
            target[action] = reward
            if not done:
                target[action] += self.params['gamma'] * max(self.Q[next_state])
            state_arr = np.array([state], dtype=int)
            target_arr = np.array([target], dtype=int)
            self.Q.model.fit(state_arr, target_arr, batch_size=batch_size, verbose=0)

    def save(self, save_dir):
        self.qnet_file = save_dir / f"{self.name}_model"
        self.Q.model.save(self.qnet_file)
        self.Q.model = None
        write_to_file(self, save_dir / f"{self.name}.pkl", text=False)
        info = (f"{type(self).__name__}\n"
                f"({self.eps_greedy.__name__}/{self.view.__name__})\n"
                f"{dict_to_str(self.params)}")
        write_to_file(info, save_dir / f"{self.name}.yml", text=True)
        print(f"Saved {self.name} to '{save_dir}'")

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.Q.model = load_model(self.qnet_file)
