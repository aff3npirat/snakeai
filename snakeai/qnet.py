from datetime import datetime

from tensorflow import keras
from tensorflow.keras import layers

from snakeai.agents import QAgent
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


class AdaptiveQnetAgent:

    def __init__(self):
        self.Q = QNet(11, 256, 4)
        self.model = AdaptiveEps(0.5, 10, 7)
        self.trainer = QNetTrainer(self.Q.model, 1.0, 0.1)
        das ist ein tets


def train():
    agent = AdaptiveQnetAgent()
    game = SnakeGame(20, 20, agent_name)

    plot_scores = []
    plot_mean_scores = []
    for k in range(1, n_episodes + 1):
        game.reset()
        done = False
        while not done:
            state = agent.get_state(game)
            action = agent.model.get_action(state, agent.Q)
            done, reward = game.play_step(action, verbosity>=2)
            next_state = agent.get_state(game)
            agent.trainer.train_step(state, action, reward, next_state, done)

            if k % 1000 == 0:
                print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{n_episodes}")

        plot_scores.append(game.score)
        plot_mean_scores.append(sum(plot_scores) / len(plot_scores))


