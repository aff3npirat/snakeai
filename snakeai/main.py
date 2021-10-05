import numpy as np
import matplotlib.pyplot as plt
import pygame
from collections import defaultdict
from datetime import datetime

from snakeai import root_dir
from snakeai.agents import EveryVisit, FirstVisit
from snakeai.model import partial_vision, diagonal_vision, full_vision, short_vision
from snakeai.model import simple_eps_decay, lin_eps_decay, constant_eps
from snakeai.helper import plot, read_from_file, save_plot
from snakeai.snake_game import SnakeGame


def train_agent(agent=None, agent_file=None, h=20, w=20, episodes=1, save=True, verbose=1):
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
    if verbose >= 1:
        plot(plot_scores, plot_mean_scores)
        save_plot(root_dir / f"agents/{agent.name}/{agent.name}.png")
    if save:
        save_dir = root_dir / f"agents/{agent.name}"
        agent.save(save_dir)
        print(f"Saved {agent.name} to '{save_dir}'")
    pygame.quit()
    plt.close()


def evaluate_monte_carlo():
    epss = np.linspace(0.1, 1.0, 10)
    gammas = np.linspace(0.0, 1.0, 11)
    ms = np.linspace(0.1, 2, 20)
    visions = {full_vision: "full",
               partial_vision: "partial",
               diagonal_vision: "diagonal",
               short_vision: "short"}

    # "vision+eps_greedy" -> "[eps, gamma, m] -> scores"
    data_ev = defaultdict(lambda: {})
    data_fv = defaultdict(lambda: {})
    for eps in epss:
        for gamma in gammas:
            for vision in visions:
                params = {"eps": eps, "gamma": gamma}
                # simple eps decay
                key1 = f"{visions[vision]}+simple"
                key2 = str([eps, gamma])
                ev_agent = EveryVisit(params, "", vision, simple_eps_decay)
                fv_agent = FirstVisit(params, "", vision, simple_eps_decay)
                train(ev_agent)
                train(fv_agent)
                data_ev[key1][key2] = evaluate(ev_agent)
                data_fv[key1][key2] = evaluate(fv_agent)

                # constant eps
                key1 = f"{visions[vision]}+const"
                ev_agent = EveryVisit(params, "", vision, constant_eps)
                fv_agent = FirstVisit(params, "", vision, constant_eps)
                train(ev_agent)
                train(fv_agent)
                data_ev[key1][key2] = evaluate(ev_agent)
                data_fv[key1][key2] = evaluate(fv_agent)

                # linear eps decay
                key1 = f"{visions[vision]}+lin"
                for m in ms:
                    key2 = str([eps, gamma, m])
                    params["m"] = m
                    ev_agent = EveryVisit(params, "", vision, lin_eps_decay)
                    fv_agent = FirstVisit(params, "", vision, lin_eps_decay)
                    train(ev_agent)
                    train(fv_agent)
                    data_ev[key1][key2] = evaluate(ev_agent)
                    data_fv[key1][key2] = evaluate(fv_agent)
    return {"ev": data_ev, "fv": data_fv}


def train(agent):
    game = SnakeGame(12, 12, False)
    scores = []
    for k in range(10_000):
        game.reset()
        agent.train_episode(game)
        scores.append(game.score)
    return scores


def evaluate(agent):
    game = SnakeGame(20, 20, False)
    scores = []
    for k in range(100):
        game.reset()
        done = False
        while not done:
            state = agent.get_state(game)
            action = agent.get_action(state)
            done, _ = game.play_step(action)
        scores.append(game.score)
    return scores
