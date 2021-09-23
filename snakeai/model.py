import math
import numpy as np


def simple_eps_decay(qs, eps, n_games):
    k = (n_games + 1) / 100
    probs = [eps / k, eps / k, eps / k, eps / k]
    probs[np.argmax(qs)] = 1 - eps / k
    return probs


def lin_eps_decay(qs, eps, m, n_games):
    y_intersect = 50 / eps
    chance = (-m * n_games + y_intersect) / y_intersect
    probs = [chance, chance, chance, chance]
    probs[np.argmax(qs)] = 1 - chance
    return probs


class AdaptiveEps:
    """Implementation based on 'Adaptive implementation of e-greedy in Reinforcment Learning' (Dos Santos Mignon, 2017).

    For readability purposes parameter l is replaced with p.
    """

    def __init__(self):
        self.max_prev = 0
        self.k = 0

    def __call__(self, qs, state, eps, p, f):
        greedy_action = np.argmax(qs)
        if np.random.uniform(0, 1) <= eps:
            max_curr = qs[greedy_action]
            self.k += 1
            new_eps = eps
            if self.k == p:
                diff = (max_curr - self.max_prev) * f
                if diff > 0:
                    new_eps = 1 / (1 + math.exp(-2 * diff)) - 0.5
                elif diff < 0:
                    new_eps = 0.5
                self.max_prev = max_curr
                self.k = 0
            return [1, 1, 1, 1], new_eps
        probs = [0, 0, 0, 0]
        probs[greedy_action] = 1
        return probs, eps
