import click

from src.Q_models import AdaptiveEpsGreedy, TDLambda
from src.agents import str_to_agent
from src.train import train


@click.command()
@click.option("-a",
              "agent_",
              type=click.Choice(["QAgent"]),
              help="Determines the agent class to use.")
@click.option("--name",
              "agent_name",
              type=str,
              help="Determines name of files, plot, game window.")
@click.option("-l",
              "--lambda",
              "lambda_",
              type=float,
              help="See TD(lambda)-algorithm for more information.")
@click.option("-lr",
              "learning_rate",
              type=float,
              help="Learning rate used.")
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
@click.option("-p",
              type=int,
              help="See 'An Adaptive Implementation of ϵ-Greedy in Reinforcement Learning' (Dos Santos Mignon, 2017).")
@click.option("-f",
              type=float,
              help="See 'An Adaptive Implementation of ϵ-Greedy in Reinforcement Learning' (Dos Santos Mignon, 2017).")
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
def run(agent_, agent_name, lambda_, learning_rate, eps, y, p, f, n_episodes, width, height, verbosity, save):
    model = AdaptiveEpsGreedy(eps, y, p, f)
    agent = str_to_agent[agent_](model)
    trainer = TDLambda(model, lambda_, learning_rate)
    train(agent, trainer, agent_name, width, height, n_episodes, verbosity, save)


if __name__ == '__main__':
    run()
