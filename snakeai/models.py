import math
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from snakeai.base import ModelBase


class TDTrainer:

    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma

    def train_step(self, state, action, reward, next_state, next_action):
        target = reward + self.gamma * self.model.Q[next_state][next_action]
        delta = target - self.model.Q[state][action]
        self.model.Q[state][action] += self.lr * delta


class FVMCTrainer:

    def __init__(self, model, gamma, visit_counter):
        self.model = model
        self.gamma = gamma
        self.visit_counter = visit_counter

    def train_step(self, episode):
        G = 0
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = self.gamma * G + reward

            if (state, action) not in [(s, a) for s, a, _ in episode[0:i]]:
                self.visit_counter[state][action] += 1
                self.model.Q[state][action] += (G - self.model.Q[state][action]) / self.visit_counter[state][action]


class SimpleEpsDecay(ModelBase):

    def __init__(self, eps):
        super().__init__(eps=eps, n_games=0)

    def get_action(self, world_state, Q):
        k = (self.n_games + 1) / 100
        if random.random() < self.eps/k or world_state not in Q:
            return random.choice([0, 1, 2, 3])
        return np.argmax(Q[world_state])


class LinEpsDecay(ModelBase):
    """Chance of doing random action decreases linear with number of games played."""

    def __init__(self, eps, m):
        super().__init__(eps=eps, m=m, n_games=0)
        self.Q = {}

    def get_action(self, world_state):
        y_intersect = 50/self.eps
        chance = (-self.m * self.n_games + y_intersect)/y_intersect
        if random.random() < chance or world_state not in self.Q:
            return random.choice([0, 1, 2, 3])
        return np.argmax(self.Q[world_state])


class AdaptiveEps(ModelBase):
    """Implementation based on 'Adaptive implementation of e-greedy in Reinforcment Learning' (Dos Santos Mignon, 2017).

    For readability purposes parameter l is replaced with p.
    """

    def __init__(self, eps, p, f):
        super().__init__(eps=eps, p=p, f=f, n_games=0)
        self.Q = {}
        self.max_prev = 0
        self.k = 0

    # noinspection PyAttributeOutsideInit
    def get_action(self, world_state):
        if world_state not in self.Q:
            self.Q[world_state] = [0.0, 0.0, 0.0, 0.0]

        greedy_action = np.argmax(self.Q[world_state])
        if np.random.uniform(0, 1) <= self.eps:
            max_curr = self.Q[world_state][greedy_action]
            self.k += 1
            if self.k == self.p:
                diff = (max_curr - self.max_prev) * self.f
                if diff > 0:
                    self.eps = 1/(1 + math.exp(-2 * diff)) - 0.5
                elif diff < 0:
                    self.eps = 0.5
                self.max_prev = max_curr
                self.k = 0
            return random.choice([0, 1, 2, 3])
        return greedy_action


class QNet(ModelBase):

    def __init__(self, input_size, hidden_size, output_size, lr):
        super().__init__(lr=lr)
        self.dense_net = keras.Sequential(
            [
                layers.Dense(hidden_size, activation="relu", input_shape=(input_size,)),
                layers.Dense(output_size, activation=None)
            ]
        ).compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                  loss=keras.losses.MeanSquaredError())

    def get_action(self, world_state):
        return np.argmax(self.dense_net(world_state))

    def __call__(self, *args, **kwargs):
        return self.dense_net(*args, **kwargs)


def get_model_by_string(string):
    model_cls = {
        "lin": LinEpsDecay,
        "adaptive": AdaptiveEps,
        "simple": SimpleEpsDecay,
        "qnet": QNet,
    }.get(string, None)
    print(f"Please enter values for the following parameters:")
    args = []
    for arg in model_cls.__init__.__code__.co_varnames[1:]:
        args.append(float(input(f"{arg}: ")))
    return model_cls(*args)
