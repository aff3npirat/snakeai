from pathlib import Path

from snakeai.base import AgentBase
from snakeai.helper import array_to_byte
from snakeai.eps_greedy import TDTrainer, AdaptiveEps
from snakeai.snake_game import TILE_SIZE, SnakeGame


class AdaptiveAgent(AgentBase):

    def __init__(self, eps, p, f, lr, gamma):
        self.model = AdaptiveEps(eps, p, f)
        self.trainer = TDTrainer(self.model, lr, gamma)

    def train_step(self, state, action, reward, next_state, next_action):
        self.trainer.train_step(state, action, next_state, reward, next_action)

    def get_action(self, state):
        return self.model.get_action(state)

    def get_state(self, game):
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
        return array_to_byte(state)


def train(agent_name, n_episodes, h, w, lr, gamma, eps, p, f, verbosity, save):
    root_dir = Path(__file__).parents[1] / f"agents/td_sarsa/{agent_name}"
    if (root_dir / f"{agent_name}.pkl").is_file(): pass
        # load agents
    else:
        agent = AdaptiveAgent(eps, p, f, lr, gamma)
    game = SnakeGame(w, h, agent_name)
    plot_scores = []
    plot_mean_scores = []
    for k in range(1, n_episodes + 1):
        game.reset()


