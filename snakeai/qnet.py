import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from keras import layers
from keras.models import load_model

from snakeai import root_dir
from snakeai.base import AgentBase
from snakeai.helper import plot, save_plot, read_from_file
from snakeai.model import AdaptiveEps, lin_eps_decay, simple_eps_decay
from snakeai.snake_game import SnakeGame


class QNetAgentBase(AgentBase):

    def __init__(self, Q, eps_greedy, **kwargs):
        super().__init__(Q, eps_greedy, **kwargs)

    def save(self, root, agent_name):
        self.Q.save(root / f"{agent_name}_model.h5")
        self.Q = None
        AgentBase.save(self, root, agent_name)

    def get_action(self, state):
        raise NotImplementedError


class AdaptiveQnetAgent(QNetAgentBase):

    def __init__(self, in_size, hidden_size, out_size, eps, p, f, gamma, lr):
        Q = keras.Sequential(
            [
                layers.Dense(hidden_size, activation="relu", name="hidden", input_dim=in_size),
                layers.Dense(out_size, name="out"),
            ]
        )
        super().__init__(Q, AdaptiveEps(), gamma=gamma, lr=lr, eps=eps, p=p, f=f)

    # noinspection PyAttributeOutsideInit
    def get_action(self, state):
        probs, self.eps = self.eps_greedy(self.Q(state), self.eps, self.p, self.f)
        return random.choices([0, 1, 2, 3], weights=probs)[0]


class SimpleQNetAgent(QNetAgentBase):

    def __init__(self, in_size, hidden_size, out_size, eps, gamma, lr):
        Q = keras.Sequential(
            [
                layers.Dense(hidden_size, activation="relu", name="hidden", input_dim=in_size),
                layers.Dense(out_size, name="out"),
            ]
        )
        # TODO: can pickle save attribute-functions?
        super().__init__(Q, simple_eps_decay, eps=eps, gamma=gamma, lr=lr)

    def get_action(self, state):
        probs = self.eps_greedy(self.Q(state), self.eps, self.n_games)
        return random.choices([0, 1, 2, 3], weights=probs)[0]


class LinQNetAgent(QNetAgentBase):

    def __init__(self, in_size, hidden_size, out_size, eps, m, gamma, lr):
        Q = keras.Sequential(
            [
                layers.Dense(hidden_size, activation="relu", name="hidden", input_dim=in_size),
                layers.Dense(out_size, name="out"),
            ]
        )
        super().__init__(Q, lin_eps_decay, gamma=gamma, lr=lr, m=m, eps=eps)

    def get_action(self, state):
        probs = self.eps_greedy(self.Q(state), self.eps, self.m, self.n_games)
        return random.choices([0, 1, 2, 3], weights=probs)[0]


# TODO: implement sarsa, which runs better?
def q_learning(agent, agent_name, h, w, n_episodes, save, verbosity):
    agent_root = root_dir / f"agents/qnet/{agent_name}"
    if (agent_root / f"{agent_name}.pkl").is_file():
        agent = read_from_file(agent_root / f"{agent_name}.pkl")
        agent.Q = load_model(agent_root / f"{agent_name}_model.h5")
        print(f"Loaded agent {agent_name}")
    game = SnakeGame(w, h)

    plot_scores = []
    plot_mean_scores = []
    for k in range(1, n_episodes + 1):
        game.reset()
        done = False
        state = agent.get_state(game)
        while not done:
            action = agent.get_action(state)
            done, reward = game.play_step(action, verbosity>=2)
            next_state = agent.get_state(game)
            # train step
            target = tf.unstack(agent.Q[state][0])
            if done:
                target[action] = reward
            else:
                target[action] = reward + agent.gamma * max(agent.Q[next_state][0])
            agent.Q.fit(np.expand_dims(state, axis=0), np.array([target]), verbose=0)
            state = next_state
        agent.n_games += 1

        # plot
        plot_scores.append(game.score)
        plot_mean_scores.append(sum(plot_scores) / len(plot_scores))
        if k % 1000 == 0:
            print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{n_episodes}")
        if verbosity >= 1:
            plot(plot_scores, plot_mean_scores)
    # save
    plot(plot_scores, plot_mean_scores)
    save_plot(agent_root / f"{agent_name}.png")
    if save:
        agent.save(agent_root, agent_name)
        print(f"Saved agent {agent_name}")
    game.quit()
