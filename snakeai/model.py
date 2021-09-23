import math
import numpy as np

from snakeai.snake_game import TILE_SIZE


# game to state functions
def short_sighted(game):
    dir_u = game.direction.value == 0  # Direction.UP
    dir_d = game.direction.value == 1  # Direction.DOWN
    dir_l = game.direction.value == 2  # Direction.LEFT
    dir_r = game.direction.value == 3  # Direction.RIGHT

    pos_u = [game.head_position[0], game.head_position[1] - TILE_SIZE]
    pos_d = [game.head_position[0], game.head_position[1] + TILE_SIZE]
    pos_l = [game.head_position[0] - TILE_SIZE, game.head_position[1]]
    pos_r = [game.head_position[0] + TILE_SIZE, game.head_position[1]]

    state = (
        # Danger straight
        (dir_u and game.is_collision(pos_u)) or
        (dir_d and game.is_collision(pos_d)) or
        (dir_l and game.is_collision(pos_l)) or
        (dir_r and game.is_collision(pos_r)),

        # Danger left
        (dir_u and game.is_collision(pos_l)) or
        (dir_d and game.is_collision(pos_r)) or
        (dir_l and game.is_collision(pos_d)) or
        (dir_r and game.is_collision(pos_u)),

        # Danger right
        (dir_u and game.is_collision(pos_r)) or
        (dir_d and game.is_collision(pos_l)) or
        (dir_l and game.is_collision(pos_u)) or
        (dir_r and game.is_collision(pos_d)),

        # Snake direction
        dir_u,
        dir_d,
        dir_l,
        dir_r,

        # Relative food position
        game.food_position[1] < game.head_position[1],
        game.food_position[1] > game.head_position[1],
        game.food_position[0] < game.head_position[1],
        game.food_position[0] > game.head_position[0],
    )
    return state


# eps decays
def simple_eps_decay(action_values, params):
    k = (params['n_games'] + 1) / 100
    prob = params['eps'] / k
    action_probs = [prob, prob, prob, prob]
    action_probs[np.argmax(action_values)] = 1 - prob
    return action_probs


def lin_eps_decay(action_values, params):
    y_intersect = 50 / params['eps']
    prob = (-params['m'] * params['n_games'] + y_intersect) / y_intersect
    action_probs = [prob, prob, prob, prob]
    action_probs[np.argmax(action_values)] = 1 - prob
    return action_probs


class AdaptiveEps:
    """Implementation based on 'Adaptive implementation of e-greedy in Reinforcment Learning' (Dos Santos Mignon, 2017).

    For readability purposes parameter l is replaced with p.
    """

    def __init__(self):
        self.max_prev = 0
        self.k = 0

    def __call__(self, action_values, params):
        greedy_action = np.argmax(action_values)
        if np.random.uniform(0, 1) <= params['eps']:
            max_curr = action_values[greedy_action]
            self.k += 1
            if self.k == params['p']:
                diff = (max_curr - self.max_prev) * params['f']
                if diff > 0:
                    params['eps'] = 1 / (1 + math.exp(-2 * diff)) - 0.5
                elif diff < 0:
                    params['eps'] = 0.5
                self.max_prev = max_curr
                self.k = 0
            return [1, 1, 1, 1]
        action_probs = [0, 0, 0, 0]
        action_probs[greedy_action] = 1
        return action_probs
