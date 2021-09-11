from datetime import datetime
from pathlib import Path

from src.helper import plot, save_plot, read_from_binary_file, save_to_binary_file, save_string_to_file, dict_to_string
from src.snake_game import SnakeGame


def train(agent, trainer, agent_name, w, h, n_episodes, verbosity, save):
    """
    Parameters
    ----------
    agent : AgentBase
    trainer : TrainerBase
    agent_name : str
    w, h : int
        Number of tiles.
    n_episodes : int
    verbosity : int
        Verbosity of 0 equals no output. Verbosity of 1, only live plotting is enabled and for verbosity of 2 (or higher) the game will
        also be rendered.
    save : bool
    """
    root_dir = Path(__file__).parents[1] / Path("agents/monte_carlo") / agent_name
    agent_fname = agent_name + ".pkl"
    # load agent
    if (root_dir / agent_fname).is_file():
        agent, trainer = read_from_binary_file(root_dir / agent_fname)
        print(f"Loaded agent '{agent_name}'")
        for key, value in agent.model.params.items():
            print(f"    {key}: {value}")

    game = SnakeGame(w, h, agent_name)
    plot_score = []
    plot_mean_score = []
    for k in range(1, n_episodes+1):
        # play and train
        game.reset()
        episode, score = game.play_episode(agent, verbosity>=2)
        trainer.train_step(episode)

        # plot score
        plot_score.append(score)
        plot_mean_score.append(sum(plot_score) / len(plot_score))

        # output progress/plots
        if k%1000 == 0:
            print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{n_episodes}")
        if verbosity >= 1:
            plot(plot_score, plot_mean_score, agent_name)
    # save
    plot(plot_score, plot_mean_score)
    save_plot(root_dir / (agent_name + ".png"))
    if save:
        if save_to_binary_file([agent, trainer], root_dir / agent_fname):
            print(f"Saved agent to '{root_dir / agent_fname}")
        if save_string_to_file(dict_to_string(agent.model.params), root_dir / (agent_name+".yml")):
            print(f"Saved parameters to '{root_dir / (agent_name + '.yml')}'")
