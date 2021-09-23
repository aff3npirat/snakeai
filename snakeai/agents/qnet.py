import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model

from snakeai.helper import write_to_file


class QNet(keras.Sequential):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        return self(item)[0]


# TODO: QNetSarsa
class QNetLearning:

    def __init__(self, params, name):
        self.params = params
        self.Q = QNet(
            [
                layers.Dense(self.params['hidden_size'], activation="relu", name="hidden",
                             input_dim=self.params['in_size']),
                layers.Dense(self.params['out_size'], name="out"),
            ]
        )
        self.name = name
        self.qnet_file = None

    def train_episode(self, game, get_state, get_action, render):
        if self.Q is None:
            self.Q = load_model(self.qnet_file)

        game.reset()
        state = get_state(game)
        done = False
        while not done:
            action = get_action(state)
            done, reward = game.play_step(action, render)
            next_state = get_state(game)
            # output of qnet has shape (1, 4)
            target = tf.unstack(self.Q[state])
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.params['gamma'] * max(self.Q[next_state])
            self.Q.fit(np.array([state]), np.array([target]), verbose=0)
            state = next_state
        self.params['n_games'] += 1

    def save(self, root_dir):
        self.qnet_file = root_dir / f"{self.name}"
        self.Q.save(self.qnet_file)
        self.Q = None
        write_to_file(self, root_dir / f"{self.name}.pkl", text=False)
        write_to_file(self.params, root_dir / f"{self.name}.yml", text=True)
        print(f"Saved agent to {root_dir}")
