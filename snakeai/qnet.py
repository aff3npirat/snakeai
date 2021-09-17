from datetime import datetime
from pathlib import Path

import tensorflow as tf

from snakeai.agents import get_agent_class_by_string
from snakeai.helper import read_from_binary_file, read_string_from_file, plot, save_plot, save_to_binary_file, save_string_to_file, \
    dict_to_string
from snakeai.models import get_model_by_string
from snakeai.snake_game import SnakeGame


def train_step(agent, model, game, gamma, verbosity):
    state = agent.get_state(game)
    action = model.get_action(state)
    done = False
    while not done:
        done, reward = game.play_step(action, verbosity>=2)
        next_state = agent.get_state(game)
        next_action = model.get_action(next_state)

        pred = model(state)
        target = tf.identity(pred)
        Q_new = reward
        if not done:
            Q_new += gamma * max(model(next_state))
        target[action] = Q_new
        model.fit(pred, target)


def evaluate_params(): pass


def qnet(agent_, model_, gamma, agent_name, w, h, n_episodes, verbosity=0, save=False):
    root_dir = Path(__file__).parents[1] / f"agents/qnet/{agent_name}"
    if (root_dir / f"{agent_name}.pkl").is_file():
        agent = read_from_binary_file(root_dir / f"{agent_name}.pkl")
        model = agent.model
        for line in read_string_from_file(root_dir / f"{agent_name}.yml"):
            name, value = line.split(": ")
            if name == "discount":
                gamma = float(value)
        print(f"Loaded agent {agent_name}")
    else:
        model = get_model_by_string(model_)
        agent = get_agent_class_by_string(agent_)(model)

    game = SnakeGame(w, h, agent_name)
    plot_scores = []
    plot_mean_scores = []
    for k in range(1, n_episodes + 1):
        game.reset()
        train_step(agent, model, game, gamma, verbosity)

        plot_scores.append(game.score)
        plot_mean_scores.append(sum(plot_scores) / len(plot_scores))
        if k % 1000 == 0:
            print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{n_episodes}")
        if verbosity >= 1:
            plot(plot_scores, plot_mean_scores, agent_name)
        plot(plot_scores, plot_mean_scores, agent_name)
        save_plot(root_dir / f"{agent_name}.png")
        if save:
            if save_to_binary_file(agent, root_dir / f"{agent_name}.pkl"):
                print(f"Saved agent to '{root_dir / f'{agent_name}.pkl'}'")
            params_to_save = {'discount': gamma}
            params_to_save.update(model.params)
            if save_string_to_file(dict_to_string(params_to_save, sep="\n"), root_dir / f"{agent_name}.yml"):
                print(f"Saved parameters to '{root_dir / f'{agent_name}.yml'}'")

