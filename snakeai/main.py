import numpy as np
import matplotlib.pyplot as plt
import pygame
from datetime import datetime

from snakeai import root_dir
from snakeai.helper import plot, read_from_file, save_plot
from snakeai.snake_game import SnakeGame


def train(agent=None, agent_file=None, h=20, w=20, episodes=1, save=True, verbose=1):
    if agent_file is not None and (root_dir / "agents" / agent_file).is_file():
        agent = read_from_file(root_dir / "agents" / agent_file)
        print(f"Loaded agent {agent.name}")
    game = SnakeGame(w, h, verbose>=3)

    plot_scores = []
    plot_mean_scores = []

    for k in range(1, episodes + 1):
        game.reset()
        agent.train_episode(game)

        # plot
        plot_scores.append(game.score)
        plot_mean_scores.append(sum(plot_scores) / len(plot_scores))
        if verbose >= 1 and (k % 1000 == 0 or k == 1):
            print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{episodes}")
        if verbose >= 2:
            plot(plot_scores, plot_mean_scores)
    # save
    plot(plot_scores, plot_mean_scores)
    save_plot(root_dir / f"agents/{agent.name}/{agent.name}.png")
    if save:
        save_dir = root_dir / f"agents/{agent.name}"
        agent.save(save_dir)
        print(f"Saved {agent.name} to '{save_dir}'")
    pygame.quit()
    plt.close()


def evaluate_monte_carlo():
    # "vision+eps_greedy" -> "[eps, gamma, m] -> [mean_score, max_score]"
    plot_data_fv = {}
    plot_data_ev = {}
    epss = np.linspace(0.1, 1.0, 10)
    gammas = np.linspace(0.0, 1.0, 11)
    ms = np.linspace(0.1, 2, 20)
    for eps in epss:
        for gamma in gammas:
            ev_agent =
