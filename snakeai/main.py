import matplotlib.pyplot as plt
import pygame
from datetime import datetime

from snakeai import root_dir
from snakeai.helper import plot, read_from_file, save_plot
from snakeai.nn_architectures import NLinearNet
from snakeai.snake_game import SnakeGame

from snakeai.qnet import QLearning
from snakeai import model


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


def evaluate_architectures(architectures):
    diagonal_ident = {}
    diagonal_sigmo = {}
    partial_ident = {}
    partial_sigmo = {}
    for arch in architectures:
        scores_ident_d = []
        max_score_ident_d = -1
        scores_sigmo_d = []
        max_score_sigmo_d = -1
        scores_ident_p = []
        max_score_ident_p = -1
        scores_sigmo_p = []
        max_score_sigmo_p = -1
        params = {"eps": 1.0, "lr": 0.001, "gamma": 0.9}
        n_episodes = 10_000

        a_diagonal_ident = QLearning(NLinearNet(arch, use_sigmoid=False), params, "evaluate", model.diagonal_vision,
                                     model.simple_eps_decay)
        a_diagonal_sigmo = QLearning(NLinearNet(arch, use_sigmoid=True), params, "evaluate", model.diagonal_vision,
                                     model.simple_eps_decay)
        a_partial_ident = QLearning(NLinearNet(arch, use_sigmoid=False), params,
                                    model.partial_vision, model.simple_eps_decay)
        a_partial_sigmo = QLearning(NLinearNet(arch, use_sigmoid=True), params,
                                    model.partial_vision, model.simple_eps_decay)

        train_game = SnakeGame(10, 10, False)
        for i in range(n_episodes):
            train_game.reset()
            a_diagonal_ident.train_episode(train_game)
            a_diagonal_sigmo.train_episode(train_game)
            a_partial_ident.train_episode(train_game)
            a_partial_sigmo.train_episode(train_game)

        val_game = SnakeGame(20, 20, False)
        for i in range(100):
            val_game.reset()
            a_diagonal_ident.train_episode(val_game)
            scores_ident_d.append(val_game.score)
            if val_game.score > max_score_ident_d:
                max_score_ident_d = val_game.score

            val_game.reset()
            a_diagonal_sigmo.train_episode(val_game)
            scores_sigmo_d.append(val_game.score)
            if val_game.score > max_score_sigmo_d:
                max_score_sigmo_d = val_game.score

            val_game.reset()
            a_partial_ident.train_episode(val_game)
            scores_ident_p.append(val_game.score)
            if val_game.score > max_score_ident_p:
                max_score_ident_p = val_game.score

            val_game.reset()
            a_partial_sigmo.train_episode(val_game)
            scores_sigmo_p.append(val_game.score)
            if val_game.score > max_score_sigmo_p:
                max_score_sigmo_p = val_game.score

        diagonal_ident[str(arch)] = [scores_ident_d, max_score_ident_d]
        diagonal_sigmo[str(arch)] = [scores_sigmo_d, max_score_sigmo_d]
        partial_ident[str(arch)] = [scores_ident_p, max_score_ident_p]
        partial_sigmo[str(arch)] = [scores_sigmo_p, max_score_sigmo_p]
    return {'diagonal_ident': diagonal_ident,
            'diagonal_sigmo': diagonal_sigmo,
            'partial_ident': partial_ident,
            'partial_sigmo': partial_sigmo}
