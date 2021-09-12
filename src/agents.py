from src.base import AgentBase
from src.helper import array_to_byte
from src.snake_game import Direction, TILE_SIZE, SnakeGame


class QAgent(AgentBase):

    def get_state(self, game: SnakeGame):
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT

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
        return array_to_byte(state)


class MarkovAgent(AgentBase):

    def get_state(self, game):
        state = [(-1, -1) for _ in range(game.x_tiles * game.y_tiles + 1)]
        state[0] = (game.food_position[0] // TILE_SIZE, game.food_position[1] // TILE_SIZE)
        state[1] = (game.head_position[0] // TILE_SIZE, game.head_position[1] // TILE_SIZE)
        i = 2
        for pos in game.body_position:
            state[i] = (pos[0] // TILE_SIZE, pos[1] // TILE_SIZE)
            i += 1
        return tuple(state)


def get_agent_class_by_string(string):
    return {"QAgent": QAgent,
            "markov": MarkovAgent}.get(string, None)

