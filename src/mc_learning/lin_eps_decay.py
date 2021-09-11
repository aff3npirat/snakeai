import click

from src.agents import str_to_agent, QAgent
from src.Q_models import LinDecayEpsGreedy, str_to_trainer, EVMonteCarlo, FVMonteCarlo
from src.train import train


@click.command()
@click.option("-a",
              "agent_",
              type=click.Choice(["QAgent"]),
              help="Determines the agent class to use.")
@click.option("-t",
              "trainer_",
              type=click.Choice(["first", "every"]),
              help="Determines trainer class to use.")
@click.option("--name",
              "agent_name",
              type=str,
              help="Determines name of files, plot, game window.")
@click.option("-e",
              "--epsilon",
              "eps",
              type=float,
              help="Chance of doing a random action.")
@click.option("-y",
              "--discount",
              "y",
              type=float,
              help="Discount factor.")
@click.option("-m",
              "--decay",
              "m",
              type=float,
              help="Decay rate of chance to do a random action. Chance = -m*n_games + 50/e")
@click.option("-w",
              "--width",
              type=int,
              help="Number of tiles along x-axis.")
@click.option("-h",
              "--height",
              type=int,
              help="Number of tiles along y-axis.")
@click.option("-n",
              "n_episodes",
              type=int,
              help="Number of episodes to train on.")
@click.option("-v",
              "verbosity",
              count=True,
              help="")
@click.option("--save",
              is_flag=True,
              help="Pass to save agent after training.")
def run(agent_, trainer_, agent_name, eps, y, m, width, height, n_episodes, verbosity, save):
    model = LinDecayEpsGreedy(eps, y, m)
    agent = str_to_agent[agent_](model)
    trainer = str_to_trainer[trainer_](model)
    train(agent, trainer, agent_name, width, height, n_episodes, verbosity, save)


if __name__ == '__main__':
    run()
