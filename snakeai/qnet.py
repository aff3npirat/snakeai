from datetime import datetime

from tensorflow import keras
from tensorflow.keras import layers

from snakeai import root_dir
from snakeai.agents import QAgent
from snakeai.base import AgentBase
from snakeai.helper import plot, save_plot, read_from_file
from snakeai.models import AdaptiveEps, QNetTrainer
from snakeai.snake_game import SnakeGame


class QNet:

    def __init__(self, input_size, hidden_size, output_size):
        self.model = keras.Sequential(
            [
                layers.InputLayer(input_shape=(input_size,), name="layer_in"),
                layers.Dense(hidden_size, activation="relu", name="layer_hidden"),
                layers.Dense(output_size, name="layer_out"),
            ]
        )

    def __getitem__(self, state):
        return self.model(state)

    def __setitem__(self, key, value):
        pass


class AdaptiveQnetAgent(AgentBase):

    def __init__(self):
        Q = QNet(11, 256, 4)
        model = AdaptiveEps(Q, 0.5, 10, 7)
        trainer = QNetTrainer(Q, 1.0, 0.1)
        super().__init__(Q, model, trainer)


def train(agent_name, h, w, n_episodes, save, verbosity):
    if (root_dir / f"agents/qnet/{agent_name}/{agent_name}.pkl").is_file():
        agent = read_from_file(root_dir / f"agents/qnet/{agent_name}/{agent_name}.pkl")
    else:
        agent = AdaptiveQnetAgent()
    game = SnakeGame(w, h, agent_name)

    plot_scores = []
    plot_mean_scores = []
    for k in range(1, n_episodes + 1):
        game.reset()
        done = False
        state = agent.get_state(game)
        while not done:
            # train
            action = agent.model.get_action(state)
            done, reward = game.play_step(action, verbosity>=2)
            next_state = agent.get_state(game)
            agent.trainer.train_step(state, action, reward, next_state, done)
            state = next_state

        # plot
        plot_scores.append(game.score)
        plot_mean_scores.append(sum(plot_scores) / len(plot_scores))
        if k % 1000 == 0:
            print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{n_episodes}")
        if verbosity >= 1:
            plot(plot_scores, plot_mean_scores, agent_name)
    # save
    plot(plot_scores, plot_mean_scores, agent_name)
    save_plot(root_dir / f"agents/qnet/{agent_name}/{agent_name}.png")
    if save:
        agent.save(root_dir / f"agents/qnet/{agent_name}", agent_name)
        print(f"Saved agent {agent_name}")


