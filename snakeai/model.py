import numpy as np

from snakeai.snake_game import TILE_SIZE, UP, DOWN, LEFT, RIGHT


# game to state functions
def markov_property(game):
    dir_u = game.direction == UP
    dir_d = game.direction == DOWN
    dir_l = game.direction == LEFT
    dir_r = game.direction == RIGHT

    state = (dir_u, dir_d, dir_l, dir_r,
             game.food[1] < game.head[1],
             game.food[1] > game.head[1],
             game.food[0] < game.head[1],
             game.food[0] > game.head[0])

    for x in range(game.x_tiles):
        for y in range(game.y_tiles):
            if game.head[0] // TILE_SIZE == x and game.head[1] // TILE_SIZE == y:
                state += (1,)
            else:
                val = 0
                for i in range(len(game.body)):
                    point_x = game.body[i][0] // TILE_SIZE
                    point_y = game.body[i][1] // TILE_SIZE
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

    pos_u = [game.head[0], game.head[1] - TILE_SIZE]
    pos_d = [game.head[0], game.head[1] + TILE_SIZE]
    pos_l = [game.head[0] - TILE_SIZE, game.head[1]]
    pos_r = [game.head[0] + TILE_SIZE, game.head[1]]

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
        game.food[1] < game.head[1],
        game.food[1] > game.head[1],
        game.food[0] < game.head[1],
        game.food[0] > game.head[0],
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


def eps_greedy(action_values, params):
    if not any(action_values):
        return [1, 1, 1, 1]

    greedy_action = np.argmax(action_values)
    action_probs = [params['eps'] for _ in range(4)]
    action_probs[greedy_action] = 1 - params['eps']
    return action_probs
