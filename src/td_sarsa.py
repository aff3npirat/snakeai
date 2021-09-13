import copy
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from src.agents import get_agent_class_by_string
from src.helper import read_from_binary_file, read_string_from_file, plot, save_plot, save_to_binary_file, save_string_to_file, \
    dict_to_string
from src.snake_game import SnakeGame, Direction
from src.q_models import get_model_by_string


def train_step(agent, model, game, lr, gamma, verbosity):
    state = agent.get_state(game)
    action = model.get_action(state)

    done = False
    while not done:
        done, reward = game.play_step(Direction(action), verbosity >= 2)
        next_state = agent.get_state(game)
        next_action = model.get_action(next_state)

        if state not in model.Q:
            model.Q[state] = [0, 0, 0, 0]
        if next_state not in model.Q:
            model.Q[next_state] = [0, 0, 0, 0]

        delta = reward + gamma * model.Q[next_state][next_action] - model.Q[state][action]
        model.Q[state][action] += lr * delta
        state = next_state
        action = next_action
    agent.model.n_games += 1


def evaluate_params(agent, lrs, gammas, w, h, n=1000, plot_name="eval_lr_gamma"):
    agent_ = copy.deepcopy(agent)
    game = SnakeGame(w, h, "evaluate_sarsa")

    plot_mean_scores = []
    for lr in lrs:
        plot_mean_scores.append([])
        for gamma in gammas:
            scores = []
            for k in range(n):
                game.reset()
                train_step(agent_, agent_.model, game, lr, gamma, 0)
                scores.append(game.score)
            # reset agent and model
            agent_ = copy.deepcopy(agent)
            plot_mean_scores[-1].append(sum(scores) / len(scores))
    game.quit()

    plt.ioff()
    plt.clf()
    plt.xlabel("gamma")
    plt.ylabel("mean_score")
    for i in range(len(lrs)):
        plt.plot(gammas, plot_mean_scores[i], label=f"lr={round(lrs[i], ndigits=2)}")
    plt.legend(loc="upper right")
    plt.savefig(Path(__file__).parents[1] / f"plots/td_sarsa/{plot_name}.png")
    plt.show()


def td_sarsa(agent_, model_, lr, gamma,  agent_name, w, h, n_episodes, verbosity=0, save=False):
    root_dir = Path(__file__).parents[1] / f"agents/td_sarsa/{agent_name}"
    if (root_dir / f"{agent_name}.pkl").is_file():
        agent = read_from_binary_file(root_dir / f"{agent_name}.pkl")
        model = agent.model
        for line in read_string_from_file(root_dir / f"{agent_name}.yml"):
            name, value = line.split(": ")
            if name == "lr":
                lr = float(value)
            elif name == "discount":
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
        train_step(agent, model, game, lr, gamma, verbosity)

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
        params_to_save = {'lr': lr, 'discount': gamma}
        params_to_save.update(model.params)
        if save_string_to_file(dict_to_string(params_to_save, sep="\n"), root_dir / f"{agent_name}.yml"):
            print(f"Saved parameters to '{root_dir / f'{agent_name}.yml'}'")


if __name__ == '__main__':
    td_sarsa("QAgent", "simple", 0.3, 1.0, "test_sarsa", 20, 20, 10000, save=True)