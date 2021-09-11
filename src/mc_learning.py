import click
from datetime import datetime
from pathlib import Path

from src.Q_models import get_model_by_string
from src.agents import get_agent_class_by_string
from src.helper import read_from_binary_file, read_string_from_file, plot, save_plot, save_to_binary_file, save_string_to_file, \
    dict_to_string
from src.snake_game import SnakeGame


@click.command()
@click.option("-a",
              "--agent",
              "agent_",
              type=click.Choice(["QAgent"]),
              help="Determines which agent to use.")
@click.option("-m",
              "--model",
              "model_",
              type=click.Choice(["lin", "adaptive", "simple"]),
              help="Determines which model to use.")
@click.option("-y",
              "--gamma",
              type=float,
              help="Discount factor.")
@click.option("--every",
              "only_first_visit",
              is_flag=True,
              default=True,
              help="If passed every visit monte carlo is used.")
@click.option("--name",
              "agent_name",
              type=str,
              help="Name of agent. If agent with same name exists he is loaded.")
@click.option("-w",
              type=int,
              help="Number of tiles along x-axis.")
@click.option("-h",
              type=int,
              help="Number of tiles along y-axis.")
@click.option("-n",
              "--n_episodes",
              type=int,
              help="Number of episodes to train.")
@click.option("-v",
              "verbosity",
              count=True,
              help="If passed at least once, plots will be outputed. If passed twice, game will be rendered.")
@click.option("--save",
              is_flag=True,
              default=False,
              help="If passed agent will be saved.")
def main(agent_, model_, gamma, only_first_visit, agent_name, w, h, n_episodes, verbosity, save):
    root_dir = Path(__file__).parents[1] / Path("agents/monte_carlo") / agent_name
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
        # play episode
        game.reset()
        episode, score = game.play_episode(agent, verbosity>=2)

        # update Q values
        G = 0
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = gamma * G + reward

            if state not in model.Q:
                model.Q[state] = [0.0, 0.0, 0.0, 0.0]
            if state not in visit_counter:
                visit_counter[state] = [0.0, 0.0, 0.0, 0.0]

            if not only_first_visit or (state, action) in [(s, a) for s, a, _ in episode[0:i]]:
                visit_counter[state][action] += 1
                model.Q[state][action] += (G - model.Q[state][action]) / visit_counter[state][action]
        model.n_games += 1

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
    main()

