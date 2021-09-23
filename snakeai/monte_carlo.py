from datetime import datetime

from snakeai import root_dir
from snakeai.base import AgentBase
from snakeai.helper import plot, save_plot, read_from_file
from snakeai.model import LinEpsDecay, SimpleEpsDecay, AdaptiveEps
from snakeai.snake_game import SnakeGame


# TODO: add visit_counter to agent
class AdaptiveMCAgent(AgentBase):

    def __init__(self, eps, p, f, gamma):
        Q = {}
        model = AdaptiveEps(Q, eps, p, f)
        trainer = FVMCTrainer(Q, gamma, {})
        super().__init__(model, trainer)


class SimpleMCAgent(AgentBase):

    def __init__(self, eps, gamma):
        Q = {}
        model = SimpleEpsDecay(Q, eps)
        trainer = FVMCTrainer(Q, gamma, {})
        super().__init__(model, trainer)


class LinMCAgent(AgentBase):

    def __init__(self, eps, m, gamma):
        Q = {}
        model = LinEpsDecay(Q, eps, m)
        trainer = FVMCTrainer(Q, gamma, {})
        super().__init__(model, trainer)


def first_visit_mc(agent, agent_name, h, w, n_episodes, save, verbosity):
    if (root_dir / f"agents/monte_carlo/{agent_name}/{agent_name}.pkl").is_file():
        agent = read_from_file(root_dir / f"agents/monte_carlo/{agent_name}/{agent_name}.pkl")
        print(f"Loaded agent {agent_name}")
    game = SnakeGame(w, h)

    plot_scores = []
    plot_mean_scores = []
    for k in range(1, n_episodes + 1):
        # train
        game.reset()
        episode = game.play_episode(agent, verbosity>=2)
        # train step
        total_reward = 0
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            total_reward = agent.gamma * total_reward + reward
            if (state, action) not in [(s, a) for s, a, _ in episode[0:i]]:
                agent.visit_counter[state][action] += 1
                agent.Q[state][action] += (total_reward - agent.Q[state][action])\
                    / agent.visit_counter[state][action]
        agent.n_games += 1

        # plot
        plot_scores.append(game.score)
        plot_mean_scores.append(sum(plot_scores) / len(plot_scores))
        if k % 1000 == 0:
            print(f"{datetime.now().strftime('%H.%M')}: episode {k}/{n_episodes}")
        if verbosity >= 1:
            plot(plot_scores, plot_mean_scores)
    # save
    plot(plot_scores, plot_mean_scores)
    save_plot(root_dir / f"agents/monte_carlo/{agent_name}/{agent_name}.png")
    if save:
        agent.save(root_dir / f"agents/monte_carlo/{agent_name}", agent_name)
        print(f"Saved agent {agent_name}")
    game.quit()
