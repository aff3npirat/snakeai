from datetime import datetime
from pathlib import Path

from src.Q_models import get_model_by_string
from src.agents import get_agent_class_by_string
from src.helper import read_from_binary_file, read_string_from_file, plot, save_plot, save_to_binary_file, save_string_to_file, \
    dict_to_string
from src.snake_game import SnakeGame


def main(agent_str, model_str, lr, lmbda, gamma, agent_name, w, h, n_episodes, verbosity, save):
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
        model = get_model_by_string(model_str)
        agent = get_agent_class_by_string(agent_str)(model)
        E = {}

    game = SnakeGame(w, h, agent_name)
    plot_scores = []
    plot_mean_scores = []
    for k in range(1, n_episodes + 1):
        game.reset()
        done = False
        while not done:
            # play step
            state = agent.get_state(game)
            action = model.get_action(state)
            done, reward = game.play_step(action, verbosity>=2)
            next_state = agent.get_state(game)

            # update eligibility traces
            state_action = (state, action)
            if state_action not in E:
                E[state_action] = 0.0
            for key in E:
                if key == state_action:
                    E[state_action] = lmbda * gamma * E[state_action] + 1
                else:
                    E[key] *= lmbda * gamma

            # update Q values
            error = reward + gamma * max(model.Q[next_state]) - model.Q[state][action]
            model.Q[state][action] += lr * E[state_action] * error

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
                              'lr': lr}.update(model.params)
            if save_string_to_file(dict_to_string(params_to_save, sep="\n"), root_dir / f"{agent_name}.yml"):
                print(f"Saved parameters to '{root_dir / f'{agent_name}.yml'}'")


if __name__ == '__main__':
    main()
