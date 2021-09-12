import click
from datetime import datetime
from pathlib import Path

from src.Q_models import get_model_by_string
from src.agents import get_agent_class_by_string
from src.helper import read_from_binary_file, read_string_from_file, plot, save_plot, save_to_binary_file, save_string_to_file, \
    dict_to_string
from src.snake_game import SnakeGame, Direction


@click.command()
@click.option("-a",
              "--agent",
              "agent_",
              type=click.Choice(["QAgent", "markov"]),
              help="Determines which agent to use.")
@click.option("-m",
              "--model",
              "model_",
              type=click.Choice(["lin", "adaptive", "simple"]),
              help="Determines which model to use.")
@click.option("-lr",
              "lr",
              type=float,
              help="Learning rate.")
@click.option("-l",
              "--lambda",
              "lmbda",
              type=float,
              help="")
@click.option("-y",
              "--gamma",
              type=float,
              help="Discount factor.")
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
def main(agent_, model_, lr, lmbda, gamma, agent_name, w, h, n_episodes, verbosity, save):
    root_dir = Path(__file__).parents[1] / Path("agents/td") / agent_name
    # load agent (if existing)
    if (root_dir / f"{agent_name}.pkl").is_file():
        agent, E = read_from_binary_file(root_dir / f"{agent_name}.pkl")
        for line in read_string_from_file(root_dir / f"{agent_name}.yml"):
            name, value = line.split(':')
            name = name.strip()
            value = value.strip()
            if name == "lambda":
                lmbda = float(value)
            elif name == "discount":
                gamma = float(value)
            elif name == "lr":
                lr = float(value)
        print(f"Loaded agent {agent_name}")
        model = agent.model
    else:
        model = get_model_by_string(model_)
        agent = get_agent_class_by_string(agent_)(model)
        E = {}

    game = SnakeGame(w, h, agent_name)
    plot_scores = []
    plot_mean_scores = []
    for k in range(1, n_episodes + 1):
        game.reset()
        state = agent.get_state(game)
        done = False
        while not done:
            # play step
            action = model.get_action(state)
            done, reward = game.play_step(Direction(action), verbosity>=2)
            next_state = agent.get_state(game)

            # update eligibility traces
            if state not in E:
                E[state] = [0.0, 0.0, 0.0, 0.0]
            if next_state not in E:
                E[next_state] = [0.0, 0.0, 0.0, 0.0]
            for key in E:
                for i in [0, 1, 2, 3]:
                    E[key][i] *= lmbda * gamma
            E[state][action] += 1

            # update Q values
            if state not in model.Q:
                model.Q[state] = [0.0, 0.0, 0.0, 0.0]
            if next_state not in model.Q:
                model.Q[next_state] = [0.0, 0.0, 0.0, 0.0]
            error = reward + gamma * max(model.Q[next_state]) - model.Q[state][action]
            for key in model.Q:
                for i in [0, 1, 2, 3]:
                    model.Q[key][i] += lr * E[key][i] * error
            state = next_state
        model.n_games += 1

        plot_scores.append(game.score)
        plot_mean_scores.append(sum(plot_scores) / len(plot_scores))
        if k%1000 == 0:
            print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{n_episodes}")
        if verbosity >= 1:
            plot(plot_scores, plot_mean_scores, agent_name)
    # save
    plot(plot_scores, plot_mean_scores, agent_name)
    save_plot(root_dir / f"{agent_name}.png")
    if save:
        if save_to_binary_file([agent, E], root_dir / f"{agent_name}.pkl"):
            print(f"Saved agent to '{root_dir / f'{agent_name}.pkl'}'")
        params_to_save = {'lambda': lmbda,
                          'discount': gamma,
                          'lr': lr}
        params_to_save.update(model.params)
        if save_string_to_file(dict_to_string(params_to_save, sep="\n"), root_dir / f"{agent_name}.yml"):
            print(f"Saved parameters to '{root_dir / f'{agent_name}.yml'}'")


if __name__ == '__main__':
    main()
