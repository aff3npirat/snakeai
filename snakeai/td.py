from datetime import datetime

from snakeai import root_dir
from snakeai.base import AgentBase
from snakeai.models import AdaptiveEps, LinEpsDecay, SimpleEpsDecay, TDTrainer
from snakeai.helper import plot, save_plot, read_from_file
from snakeai.snake_game import SnakeGame


class AdaptiveTDAgent(AgentBase):

    def __init__(self, eps, p, f, gamma, lr):
        Q = {}
        model = AdaptiveEps(Q, eps, p, f)
        trainer = TDTrainer(Q, gamma, lr)
        super().__init__(model, trainer)


class SimpleTDAgent(AgentBase):

    def __init__(self, eps, gamma, lr):
        Q = {}
        model = SimpleEpsDecay(Q, eps)
        trainer = TDTrainer(Q, gamma, lr)
        super().__init__(model, trainer)


class LinTDAgent(AgentBase):

    def __init__(self, eps, m, gamma, lr):
        Q = {}
        model = LinEpsDecay(Q, eps, m)
        trainer = TDTrainer(Q, gamma, lr)
        super().__init__(model, trainer)


def train(agent, agent_name, h, w, n_episodes, save, verbosity):
    if (root_dir / f"agents/TD/{agent_name}/{agent_name}.pkl").is_file():
        agent = read_from_file(root_dir / f"agents/TD/{agent_name}/{agent_name}.pkl")
    game = SnakeGame(w, h)

    plot_scores = []
    plot_mean_scores = []
    for k in range(1, n_episodes + 1):
        game.reset()
        state = agent.get_state(game)
        action = agent.model.get_action(state)
        done = False
        while not done:
            # train
            reward, done = game.play_step(action, verbosity>=2)
            next_state = agent.get_state(game)
            next_action = agent.model.get_action(state)
            agent.trainer.train_step(state, action, reward, next_state, next_action)
            agent.model.n_games += 1

            # plot
            plot_scores.append(game.score)
            plot_mean_scores.append(sum(plot_scores) / len(plot_scores))
            if k % 1000 == 0:
                print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{n_episodes}")
            if verbosity >= 1:
                plot(plot_scores, plot_mean_scores)
    # save
    plot(plot_scores, plot_mean_scores)
    save_plot(root_dir / f"agents/TD/{agent_name}/{agent_name}.png")
    if save:
        agent.save(root_dir / f"agents/TD/{agent_name}", agent_name)
        print(f"Saved agent {agent_name}")