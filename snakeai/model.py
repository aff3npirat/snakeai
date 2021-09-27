import math
import numpy as np

from snakeai.snake_game import TILE_SIZE, UP, DOWN, LEFT, RIGHT


# game to state functions
def markov_property(game):
    dir_u = game.direction == UP
    dir_d = game.direction == DOWN
    dir_l = game.direction == LEFT
    dir_r = game.direction == RIGHT

    state = (dir_u, dir_d, dir_l, dir_r,
             game.food_position[1] < game.head_position[1],
             game.food_position[1] > game.head_position[1],
             game.food_position[0] < game.head_position[1],
             game.food_position[0] > game.head_position[0])

    for x in range(game.x_tiles):
        for y in range(game.y_tiles):
            if game.head_position[0] // TILE_SIZE == x and game.head_position[1] // TILE_SIZE == y:
                state += (1,)
            else:
                val = 0
                for i in range(len(game.body_position)):
                    point_x = game.body_position[i][0] // TILE_SIZE
                    point_y = game.body_position[i][1] // TILE_SIZE
                    if point_x == x and point_y == y:
                        val = 2
                        break
                state += (val,)
    return state


def short_sighted(game):
    dir_u = game.direction == UP
    dir_d = game.direction == DOWN
    dir_l = game.direction == LEFT
    dir_r = game.direction == RIGHT

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
    if not any(action_values):
        return [1, 1, 1, 1]

    greedy_action = np.argmax(action_values)
    if params['n_games'] >= 100:
        prob_explore = 0
    else:
        prob_explore = (100-params['n_games']) / 100
    action_probs = [prob_explore for _ in range(4)]
    action_probs[greedy_action] = 1 - prob_explore
    return action_probs


def lin_eps_decay(action_values, params):
    if not any(action_values):
        return [1, 1, 1, 1]

    greedy_action = np.argmax(action_values)
    prob_explore = -params['m'] * params['n_games'] + 1
    if prob_explore < 0:
        prob_explore = 0
    action_probs = [prob_explore for _ in range(4)]
    action_probs[greedy_action] = 1 - prob_explore
    return action_probs


def adaptive_eps_decay(self, action_values, params):
    greedy_action = np.argmax(action_values)
    if np.random.uniform(0, 1) <= params['eps']:
        max_curr = action_values[greedy_action]
        params['k'] += 1
        if params['k'] == params['p']:
            diff = (max_curr - self.max_prev) * params['f']
            if diff > 0:
                params['eps'] = 1 / (1 + math.exp(-2 * diff)) - 0.5
            elif diff < 0:
                params['eps'] = 0.5
            params['max_prev'] = max_curr
            params['k'] = 0
        return [1, 1, 1, 1]
    action_probs = [0, 0, 0, 0]
    action_probs[greedy_action] = 1
    return action_probs
