import matplotlib.pyplot as plt
import pygame
import random
from datetime import datetime

from snakeai import root_dir
from snakeai.helper import plot, read_from_file, save_plot
from snakeai.snake_game import SnakeGame


def train(agent=None, agent_file=None, h=20, w=20, n_episodes=1, save=True, verbose=2):
    if agent_file is not None and (root_dir / "agents" / agent_file).is_file():
        agent = read_from_file(root_dir / "agents" / agent_file)
        print(f"Loaded agent {agent.name}")
    game = SnakeGame(w, h)

    plot_scores = []
    plot_mean_scores = []
    for k in range(1, n_episodes + 1):
        agent.train_episode(game, verbose>=3)

        # plot
        plot_scores.append(game.score)
        plot_mean_scores.append(sum(plot_scores) / len(plot_scores))
        if verbose >= 1 and k % 1000 == 0:
            print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{n_episodes}")
        if verbose >= 2:
            plot(plot_scores, plot_mean_scores)
    # save
    plot(plot_scores, plot_mean_scores)
    save_plot(root_dir / f"agents/{agent.name}/{agent.name}.png")
    if save:
        agent.save(root_dir / f"agents/{agent.name}")
    pygame.quit()
    plt.close()
