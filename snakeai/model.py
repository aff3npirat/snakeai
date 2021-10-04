import numpy as np

from snakeai.snake_game import TILE_SIZE, UP, DOWN, LEFT, RIGHT


# game to state functions
def partial_vision(game):
    if min(game.x_tiles, game.y_tiles) < 11:
        raise ValueError("board size should be greater equal 11")

    state = []
    for x in range(-5, 6):
        for y in range(-5, 6):
            pos = [game.head[0] + x*TILE_SIZE, game.head[1] + y*TILE_SIZE]
            if pos == game.food:
                state.append(3)
            elif pos == game.head:
                state.append(1)
            elif pos in game.body:
                state.append(2)
            elif game.out_of_bounds(pos):
                state.append(-1)
            else:
                state.append(0)
    return tuple(state)


def diagonal_vision(game):
    state = (
        game.direction == UP,
        game.direction == DOWN,
        game.direction == LEFT,
        game.direction == RIGHT,
    )

    x, y = game.head

    def diagonals(step):
        d1 = [x, y - step * TILE_SIZE]  # up
        d2 = [x, y + step * TILE_SIZE]  # down
        d3 = [x - step * TILE_SIZE, y]  # left
        d4 = [x + step * TILE_SIZE, y]  # right
        d5 = [x + step * TILE_SIZE, y + step * TILE_SIZE]  # bottom right
        d6 = [x + step * TILE_SIZE, y - step * TILE_SIZE]  # top right
        d7 = [x - step * TILE_SIZE, y + step * TILE_SIZE]  # bottom left
        d8 = [x - step * TILE_SIZE, y - step * TILE_SIZE]  # top left
        return [d1, d2, d3, d4, d5, d6, d7, d8]

    wall_distances = [0 for _ in range(8)]
    food_distances = [0 for _ in range(8)]
    body_distances = [0 for _ in range(8)]
    for i in range(1, max(game.x_tiles, game.y_tiles)):
        diagonals_ = diagonals(i)
        for j in range(8):
            if not game.out_of_bounds(diagonals_[j]):
                wall_distances[j] += 1
            if diagonals_[j] != game.food:
                food_distances[j] += 1
            if not diagonals_[j] in game.body:
                body_distances[j] += 1
    state += tuple(wall_distances)
    state += tuple(food_distances)
    state += tuple(body_distances)
    return state


# TODO: performance when seeing head as normal body
def full_vision(game):
    state = (game.direction == UP,
             game.direction == DOWN,
             game.direction == LEFT,
             game.direction == RIGHT,
             game.food[1] < game.head[1],
             game.food[1] > game.head[1],
             game.food[0] < game.head[1],
             game.food[0] > game.head[0])

    for x in range(game.x_tiles):
        for y in range(game.y_tiles):
            pos = [x * TILE_SIZE, y * TILE_SIZE]
            if pos == game.head:
                state += (1,)
            elif pos in game.body:
                state += (2,)
            else:
                state += (0,)
    return state


def short_vision(game):
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


def constant_eps(action_values, params):
    if not any(action_values):
        return [1, 1, 1, 1]

    greedy_action = np.argmax(action_values)
    action_probs = [params['eps'] for _ in range(4)]
    action_probs[greedy_action] = 1 - params['eps']
    return action_probs
