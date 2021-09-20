import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from snakeai.base import ModelBase, TrainerBase


class TDTrainer(TrainerBase):

    def __init__(self, Q, gamma, lr):
        super().__init__(Q, gamma=gamma, lr=lr)

    def train_step(self, state, action, reward, next_state, next_action):
        # TODO: target = reward + gamma * max(Q[next_state]), which runs better?
        target = reward + self.gamma * self.Q[next_state][next_action]
        delta = target - self.Q[state][action]
        self.Q[state][action] += self.lr * delta


class FVMCTrainer(TrainerBase):

    def __init__(self, Q, gamma, visit_counter):
        super().__init__(Q, gamma=gamma)
        self.visit_counter = visit_counter

    def train_step(self, episode):
        G = 0
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = self.gamma * G + reward

            if (state, action) not in [(s, a) for s, a, _ in episode[0:i]]:
                self.visit_counter[state][action] += 1
                self.Q[state][action] += (G - self.Q[state][action]) / self.visit_counter[state][action]


class QNetTrainer(TrainerBase):

    def __init__(self, Q, gamma, lr):
        Q.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                        loss=keras.losses.MeanSquaredError())
        super().__init__(Q, gamma=gamma, lr=lr)

    def train_step(self, state, action, reward, next_state, done):
        pred = self.Q[state]
        target = tf.identity(pred)
        if done:
            target[action] = reward
        else:
            # TODO: target[action] = reward + gamma * Q[next_state][next_action], which runs better?
            target[action] = reward + self.gamma * max(self.Q[next_state])
        self.Q.model.fit(state, target)


class SimpleEpsDecay(ModelBase):

    def __init__(self, Q, eps):
        super().__init__(Q, eps=eps, n_games=0)

    def get_action(self, world_state):
        k = (self.n_games + 1) / 100
        if random.random() < self.eps/k or world_state not in self.Q:
            return random.choice([0, 1, 2, 3])
        return np.argmax(self.Q[world_state])


class LinEpsDecay(ModelBase):
    """Chance of doing random action decreases linear with number of games played."""

    def __init__(self, Q, eps, m):
        super().__init__(Q, eps=eps, m=m, n_games=0)

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

    def __init__(self, Q, eps, p, f):
        super().__init__(Q, eps=eps, p=p, f=f, n_games=0)
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


def get_model_by_string(string):
    model_cls = {
        "lin": LinEpsDecay,
        "adaptive": AdaptiveEps,
        "simple": SimpleEpsDecay,
    }.get(string, None)
    print(f"Please enter values for the following parameters:")
    args = []
    for arg in model_cls.__init__.__code__.co_varnames[1:]:
        args.append(float(input(f"{arg}: ")))
    return model_cls(*args)
