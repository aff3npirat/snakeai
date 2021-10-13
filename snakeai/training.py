import numpy as np
import matplotlib.pyplot as plt
import pygame
from collections import defaultdict
from datetime import datetime

from . import agents
from . import helpers
from . import model
from .snake_game import SnakeGame


def train_agent(agent=None,
                h=20,
                w=20,
                episodes=1,
                save=True,
                verbose=1
                ):
    """
    Parameters
    ----------
    agent : QAgentBase or str
    h : int
    w : int
    episodes : int
    save : bool
    verbose : int
    """
    top_level = helpers.get_top_directory()
    if isinstance(agent, str) and (top_level / agent).is_file():
        agent = helpers.read_from_file(agent)
        print(f"Loaded agent {agent.name}")
    game = SnakeGame(w, h, verbose >= 3)

    plot_scores = []
    plot_mean_scores = []
    for k in range(1, episodes + 1):
        game.reset()
        agent.train_episode(game)

        if save and game.score > agent.params['record']:
            save_dir = f"agents/{agent.name}"
            agent.save(save_dir)
            print(f"Saved agent to '{save_dir}'")

        # plot
        plot_scores.append(game.score)
        plot_mean_scores.append(sum(plot_scores) / len(plot_scores))
        if verbose >= 1 and (k % 1000 == 0 or k == 1):
            print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{episodes}")
        if verbose >= 2:
            helpers.plot_scores(plot_scores, plot_mean_scores)
    # save
    if verbose >= 1:
        helpers.plot_scores(plot_scores, plot_mean_scores)
        helpers.save_plot(f"agents/{agent.name}/{agent.name}.png")
    if save:
        save_dir = f"agents/{agent.name}"
        agent.save(save_dir)
        print(f"Saved agent to '{save_dir}'")
    pygame.quit()
    plt.close()


def evaluate_monte_carlo():
    epss = np.linspace(0.1, 1.0, 10)
    gammas = np.linspace(0.0, 1.0, 11)
    ms = np.linspace(0.1, 2, 20)
    visions = {model.full_vision: "full",
               model.partial_vision: "partial",
               model.diagonal_vision: "diagonal",
               model.short_vision: "short"}

    # "vision+eps_greedy" -> "[eps, gamma, m] -> scores"
    data_ev = defaultdict(lambda: {})
    data_fv = defaultdict(lambda: {})
    # initial call to print 0% progress
    i = 0
    helpers.print_progress_bar(i,
                               440,
                               prefix="Progress:",
                               suffix="complete",
                               length=80)
    for eps in epss:
        for gamma in gammas:
            for vision in visions:
                params = {"eps": eps, "gamma": gamma}
                # simple eps decay
                key1 = f"{visions[vision]}+simple"
                key2 = str([eps, gamma])
                ev_agent = agents.EveryVisit(params,
                                             "",
                                             vision,
                                             model.simple_eps_decay)
                fv_agent = agents.FirstVisit(params,
                                             "",
                                             vision,
                                             model.simple_eps_decay)
                train(ev_agent)
                train(fv_agent)
                data_ev[key1][key2] = evaluate(ev_agent)
                data_fv[key1][key2] = evaluate(fv_agent)

                # constant eps
                key1 = f"{visions[vision]}+const"
                ev_agent = agents.EveryVisit(params,
                                             "",
                                             vision,
                                             model.constant_eps)
                fv_agent = agents.FirstVisit(params,
                                             "",
                                             vision,
                                             model.constant_eps)
                train(ev_agent)
                train(fv_agent)
                data_ev[key1][key2] = evaluate(ev_agent)
                data_fv[key1][key2] = evaluate(fv_agent)

                # linear eps decay
                key1 = f"{visions[vision]}+lin"
                for m in ms:
                    key2 = str([eps, gamma, m])
                    params["m"] = m
                    ev_agent = agents.EveryVisit(params,
                                                 "",
                                                 vision,
                                                 model.lin_eps_decay)
                    fv_agent = agents.FirstVisit(params,
                                                 "",
                                                 vision,
                                                 model.lin_eps_decay)
                    train(ev_agent)
                    train(fv_agent)
                    data_ev[key1][key2] = evaluate(ev_agent)
                    data_fv[key1][key2] = evaluate(fv_agent)
                i += 1
                helpers.print_progress_bar(i,
                                           440,
                                           prefix="Progress:",
                                           suffix="complete",
                                           length=80)
    return {"ev": dict(data_ev), "fv": dict(data_fv)}


def evaluate_td():
    lrs = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
           1.0]
    epss = np.linspace(0.1, 1.0, 10)
    gammas = np.linspace(0.0, 1.0, 11)
    ms = np.linspace(0.1, 2, 20)
    visions = {model.full_vision: "full",
               model.partial_vision: "partial",
               model.diagonal_vision: "diagonal",
               model.short_vision: "short"}
    sarsa = defaultdict(lambda: {})
    qlearning = defaultdict(lambda: {})
    i = 0
    helpers.print_progress_bar(i,
                               5720,
                               prefix="Progress:",
                               suffix="complete",
                               length=80)
    for lr in lrs:
        for eps in epss:
            for gamma in gammas:
                for vision in visions:
                    p = {"eps": eps, "gamma": gamma, "lr": lr}
                    # simple
                    model_id = f"{visions[vision]}+simple"
                    param_id = str([eps, gamma, lr])
                    sarsa_agent = agents.TDSarsa(p,
                                                 "",
                                                 vision,
                                                 model.simple_eps_decay)
                    qlearn_agent = agents.TDQLearning(p,
                                                      "",
                                                      vision,
                                                      model.simple_eps_decay)
                    train(sarsa_agent)
                    train(qlearn_agent)
                    sarsa[model_id][param_id] = evaluate(sarsa_agent)
                    qlearning[model_id][param_id] = evaluate(qlearn_agent)
                    # constant
                    model_id = f"{visions[vision]}+const"
                    sarsa_agent = agents.TDSarsa(p,
                                                 "",
                                                 vision,
                                                 model.constant_eps)
                    qlearn_agent = agents.TDQLearning(p,
                                                      "",
                                                      vision,
                                                      model.constant_eps)
                    train(sarsa_agent)
                    train(qlearn_agent)
                    sarsa[model_id][param_id] = evaluate(sarsa_agent)
                    qlearning[model_id][param_id] = evaluate(qlearn_agent)
                    # linear
                    model_id = f"{visions[vision]}+lin"
                    for m in ms:
                        p["m"] = m
                        param_id = str([eps, gamma, lr, m])
                        sarsa_agent = agents.TDSarsa(p,
                                                     "",
                                                     vision,
                                                     model.lin_eps_decay)
                        qlearn_agent = agents.TDQLearning(p,
                                                          "",
                                                          vision,
                                                          model.lin_eps_decay)
                        train(sarsa_agent)
                        train(qlearn_agent)
                        sarsa[model_id][param_id] = evaluate(sarsa_agent)
                        qlearning[model_id][param_id] = evaluate(qlearn_agent)
                    i += 1
                    helpers.print_progress_bar(i,
                                               5720,
                                               prefix="Progress:",
                                               suffix="complete",
                                               length=80)
    return {"sarsa": dict(sarsa), "qlearning": dict(qlearning)}


# TODO
def evaluate_qnet(): pass


def train(agent):
    game = SnakeGame(12, 12, False)
    scores = []
    for k in range(1000):
        game.reset()
        agent.train_episode(game)
        scores.append(game.score)
    return scores


def evaluate(agent):
    game = SnakeGame(20, 20, False)
    scores = []
    for k in range(50):
        game.reset()
        done = False
        while not done:
            state = agent.get_state(game)
            action = agent.get_action(state)
            done, _ = game.play_step(action)
        scores.append(game.score)
    return scores
