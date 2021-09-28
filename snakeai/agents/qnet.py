import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model

from snakeai.helper import dict_to_str, write_to_file


class QNet:

    # TODO: remove in_size, keras.Seqeuntial can infer input size from first input
    def __init__(self, in_size, hidden_size, out_size, lr, loss=keras.losses.MeanSquaredError()):
        self.model = keras.Sequential([
            layers.Dense(hidden_size, activation="relu", name="hidden", input_dim=in_size),
            layers.Dense(out_size, name="out")
        ])
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)

    def __getitem__(self, state):
        return self.model(np.array([state], dtype=int))[0]


# TODO: QNetSarsa
class QNetLearning:

    def __init__(self, params, name):
        self.params = params
        self.Q = QNet(params['in_size'], params['hidden_size'], params['out_size'], params['lr'])
        self.name = name
        self.qnet_file = None

    def train_episode(self, game, get_state, get_action, render):
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
            state_arr = np.array([state], dtype=int)
            target_arr = np.array([target], dtype=float)
            self.Q.model.fit(state_arr, target_arr, verbose=0)
            state = next_state
        self.params['n_games'] += 1

    def save(self, save_dir):
        self.qnet_file = save_dir / f"{self.name}_model"
        self.Q.model.save(self.qnet_file)
        self.Q.model = None
        write_to_file(self, save_dir / f"{self.name}.pkl", text=False)
        write_to_file(dict_to_str(self.params), save_dir / f"{self.name}.yml", text=True)
        print(f"Saved {self.name} to '{save_dir}'")

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.Q.model = load_model(self.qnet_file)
