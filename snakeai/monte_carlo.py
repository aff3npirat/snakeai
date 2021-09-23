from datetime import datetime

from snakeai import root_dir
from snakeai.base import AgentBase
from snakeai.helper import plot, save_plot, read_from_file, write_to_file
from snakeai.model import AdaptiveEps, lin_eps_decay, simple_eps_decay
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
    agent_root = root_dir / f"agents/monte_carlo/{agent_name}"
    if (agent_root / f"{agent_name}.pkl").is_file():
        agent = read_from_file(agent_root / f"{agent_name}.pkl")
        num_visits = read_from_file(agent_root / "num_visit.pkl")
        print(f"Loaded agent {agent_name}")
    else:
        num_visits = {}
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
                if state not in num_visits:
                    num_visits[state] = [0, 0, 0, 0]
                num_visits[state][action] += 1
                agent.Q[state][action] += (total_reward - agent.Q[state][action])\
                    / num_visits[state][action]
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
    save_plot(agent_root / f"{agent_name}.png")
    if save:
        agent.save(agent_root, agent_name)
        write_to_file(num_visits, agent_root / "num_visits.pkl")
        print(f"Saved agent {agent_name}")
    game.quit()
