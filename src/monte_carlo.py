import copy
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from src.agents import get_agent_class_by_string
from src.models import get_model_by_string
from src.helper import read_from_binary_file, read_string_from_file, plot, save_plot, save_to_binary_file, save_string_to_file, \
    dict_to_string
from src.snake_game import SnakeGame


def evaluate_params(agent, first_visit, gammas, w, h, n=1000, plot_name="eval_gamma"):
    agent_ = copy.deepcopy(agent)
    game = SnakeGame(w, h, "evaluate_mc")
    visit_counter = {}

    plot_mean_scores = []
    for gamma in gammas:
        scores = []
        for k in range(n):
            game.reset()
            train_step(agent_, agent_.model, game, visit_counter, first_visit, gamma, 0)
            scores.append(game.score)
        agent_ = copy.deepcopy(agent)
        visit_counter = {}
        plot_mean_scores.append(sum(scores) / len(scores))
    game.quit()

    plt.ioff()
    plt.clf()
    plt.xlabel("gamma")
    plt.ylabel("mean_score")
    plt.plot(gammas, plot_mean_scores)
    plt.savefig(Path(__file__).parents[1] / f"plots/monte_carlo/{plot_name}.png")
    plt.show()


def train_step(agent, model, game, visit_counter, gamma, first_visit, verbosity):
    episode, score = game.play_episode(agent, verbosity>=2)
    G = 0
    for i in reversed(range(len(episode))):
        state, action, reward = episode[i]
        G = gamma * G + reward

        if state not in model.Q:
            model.Q[state] = [0.0, 0.0, 0.0, 0.0]
        if state not in visit_counter:
            visit_counter[state] = [0.0, 0.0, 0.0, 0.0]

        if not first_visit or (state, action) not in [(s, a) for s, a, _ in episode[0:i]]:
            visit_counter[state][action] += 1
            model.Q[state][action] += (G - model.Q[state][action]) / visit_counter[state][action]
    model.n_games += 1


def mc_learning(agent_, model_, first_visit, gamma, agent_name, w, h, n_episodes, verbosity=0, save=False):
    root_dir = Path(__file__).parents[1] / Path("agent/monte_carlo") / agent_name
    # load agent (if existing)
    if (root_dir / f"{agent_name}.pkl").is_file():
        agent, visit_counter = read_from_binary_file(root_dir / f"{agent_name}.pkl")
        for line in read_string_from_file(root_dir / f"{agent_name}.yml"):
            name, value = line.split(":")
            name = name.strip()
            value = value.strip()
            if name == "discount":
                gamma = float(value)
        print(f"Loaded agent {agent_name}")
        model = agent.model
    else:
        model = get_model_by_string(model_)
        agent = get_agent_class_by_string(agent_)(model)
        visit_counter = {}

    game = SnakeGame(w, h, agent_name)
    plot_scores = []
    plot_mean_scores = []
    for k in range(1, n_episodes + 1):
        game.reset()
        train_step(agent, model, game, visit_counter, gamma, first_visit, verbosity)

        plot_scores.append(game.score)
        plot_mean_scores.append(sum(plot_scores) / len(plot_scores))
        if k % 1000 == 0:
            print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{n_episodes}")
        if verbosity >= 1:
            plot(plot_scores, plot_mean_scores, agent_name)
    # save
    plot(plot_scores, plot_mean_scores, agent_name)
    save_plot(root_dir / f"{agent_name}.png")
    if save:
        if save_to_binary_file([agent, visit_counter], root_dir / f"{agent_name}.pkl"):
            print(f"Saved agent to '{root_dir / f'{agent_name}.pkl'}'")
        params_to_save = {'discount': gamma}
        params_to_save.update(model.params)
        if save_string_to_file(dict_to_string(params_to_save, sep="\n"), root_dir / f"{agent_name}.yml"):
            print(f"Saved parameters to '{root_dir / f'{agent_name}.yml'}'")


if __name__ == '__main__':
    mc_learning("QAgent", "simple", False, 1.0, "test_mc", 20, 20, 10000, save=True)

