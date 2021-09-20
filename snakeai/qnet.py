from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers

from snakeai import root_dir
from snakeai.base import AgentBase
from snakeai.helper import plot, save_plot, read_from_file
from snakeai.models import AdaptiveEps, QNetTrainer, SimpleEpsDecay, LinEpsDecay
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

    def __init__(self, in_size, hidden_size, out_size, eps, p, f, gamma, lr):
        Q = QNet(in_size, hidden_size, out_size)
        model = AdaptiveEps(Q, eps, p, f)
        trainer = QNetTrainer(Q, gamma, lr)
        super().__init__(model, trainer)


class SimpleQNetAgent(AgentBase):

    def __init__(self, in_size, hidden_size, out_size, eps, gamma, lr):
        Q = QNet(in_size, hidden_size, out_size)
        model = SimpleEpsDecay(Q, eps)
        trainer = QNetTrainer(Q, gamma, lr)
        super().__init__(model, trainer)


class LinQNetAgent(AgentBase):

    def __init__(self, in_size, hidden_size, out_size, eps, m, gamma, lr):
        Q = QNet(in_size, hidden_size, out_size)
        model = LinEpsDecay(Q, eps, m)
        trainer = QNetTrainer(Q, gamma, lr)
        super().__init__(model, trainer)


def train(agent, agent_name, h, w, n_episodes, save, verbosity):
    if (root_dir / f"agents/qnet/{agent_name}/{agent_name}.pkl").is_file():
        agent = read_from_file(root_dir / f"agents/qnet/{agent_name}/{agent_name}.pkl")
    game = SnakeGame(w, h)

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
            agent.model.n_games += 1
            state = next_state

        # plot
        plot_scores.append(game.score)
        plot_mean_scores.append(sum(plot_scores) / len(plot_scores))
        if k % 1000 == 0:
            print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{n_episodes}")
        if verbosity >= 1:
            plot(plot_scores, plot_mean_scores)
    # save
    plot(plot_scores, plot_mean_scores)
    save_plot(root_dir / f"agents/qnet/{agent_name}/{agent_name}.png")
    if save:
        agent.save(root_dir / f"agents/qnet/{agent_name}", agent_name)
        print(f"Saved agent {agent_name}")
