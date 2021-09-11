import math
import random
import numpy as np

from src.base import TrainerBase, ModelBase


# Models
class QModel(ModelBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Q = {}
        self.state_action_counter = {}

    def get_action(self, world_state):
        raise NotImplementedError


class LinDecayEpsGreedy(QModel):
    """Chance of doing random action decreases linear with number of games played."""

    def __init__(self, eps, y, m):
        super().__init__(eps=eps, y=y, m=m, n_games=0)

    def get_action(self, world_state):
        y_intersect = 50/self.eps
        chance = (-self.m * self.n_games + y_intersect)/y_intersect
        if random.random() < chance or world_state not in self.Q:
            return random.choice([0, 1, 2, 3])
        return np.argmax(self.Q[world_state])


class AdaptiveEpsGreedy(QModel):
    """Implementation based on 'Adaptive implementation of e-greedy in Reinforcment Learning' (Dos Santos Mignon, 2017).

    For readability purposes parameter l is replaced with p.
    """

    def __init__(self, eps, y, p, f):
        super().__init__(eps=eps, y=y, p=p, f=f, n_games=0)
        self.max_prev = 0
        self.k = 0

    # noinspection PyAttributeOutsideInit
    def get_action(self, world_state):
        if world_state not in self.Q:
            self.Q[world_state] = [0.0, 0.0, 0.0, 0.0]
            self.state_action_counter[world_state] = [0, 0, 0, 0]

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


# Trainers
# Monte Carlo
class FVMonteCarlo(TrainerBase):

    def train_step(self, episode):
        self.model.n_games += 1
        self._init_Q(episode)
        Q = self.model.Q
        visit_counter = self.model.state_action_counter
        G = 0
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = self.model.y * G + reward
            if (state, action) not in [(s, a) for s, a, _ in episode[:i]]:
                if visit_counter[state][action] == 0:
                    Q[state][action] = G
                else:
                    n = visit_counter[state][action]
                    Q[state][action] += (G-Q[state][action]) / (n+1)
                visit_counter[state][action] += 1

    def _init_Q(self, episode):
        for s, _, _ in episode:
            if s not in self.model.Q:
                self.model.Q[s] = [0.0, 0.0, 0.0, 0.0]
                self.model.state_action_counter[s] = [0, 0, 0, 0]


class EVMonteCarlo(TrainerBase):

    def train_step(self, episode):
        self.model.n_games += 1
        self._init_Q(episode)
        Q = self.model.Q
        visit_counter = self.model.state_action_counter
        G = 0
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = self.model.y * G + reward
            if visit_counter[state][action] == 0:
                Q[state][action] = G
            else:
                n = visit_counter[state][action]
                Q[state][action] += (G-Q[state][action]) / (n+1)
            visit_counter[state][action] += 1

    def _init_Q(self, episode):
        for s, _, _ in episode:
            if s not in self.model.Q:
                self.model.Q[s] = [0.0, 0.0, 0.0, 0.0]
                self.model.state_action_counter[s] = [0, 0, 0, 0]


class TDLambda(TrainerBase):

    def __init__(self, model):
        super().__init__(model)
        # keeps track of eligibility trace for every state-action pair
        self.E = {}

    def train_step(self, episode):
        pass


str_to_trainer = {'first': FVMonteCarlo,
                  'every': EVMonteCarlo}
