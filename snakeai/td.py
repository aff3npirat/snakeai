import random
from datetime import datetime

from snakeai import root_dir
from snakeai.base import AgentBase
from snakeai.model import AdaptiveEps, lin_eps_decay, simple_eps_decay
from snakeai.helper import plot, save_plot, read_from_file
from snakeai.snake_game import SnakeGame


class AdaptiveTDAgent(AgentBase):

    def __init__(self, eps, p, f, gamma, lr):
        super().__init__({}, AdaptiveEps(), eps=eps, p=p, f=f, gamma=gamma, lr=lr)

    # noinspection PyAttributeOutsideInit
    def get_action(self, state):
        if state not in self.Q:
            self.Q[state] = [[0], [0], [0], [0]]
        probs, self.eps = self.eps_greedy(self.Q[state], state, self.eps, self.p, self.f)
        return random.choices([0, 1, 2, 3], weights=probs)


class SimpleTDAgent(AgentBase):

    def __init__(self, eps, gamma, lr):
        super().__init__({}, simple_eps_decay, eps=eps, gamma=gamma, lr=lr)

    def get_action(self, state):
        if state not in self.Q:
            self.Q[state] = [0, 0, 0, 0]
        probs = self.eps_greedy(self.Q, state, self.eps, self.n_games)
        return random.choices([0, 1, 2, 3], weights=probs)


class LinTDAgent(AgentBase):

    def __init__(self, eps, m, gamma, lr):
        super().__init__({}, lin_eps_decay, eps=eps, m=m, gamma=gamma, lr=lr)

    def get_action(self, state):
        if state not in self.Q:
            self.Q[state] = [0, 0, 0, 0]
        probs = self.eps_greedy(self.Q, state, self.eps, self.m, self.n_games)
        return random.choices([0, 1, 2, 3], weights=probs)


# TODO: implement q-learning, which runs better?
def sarsa(agent, agent_name, h, w, n_episodes, save, verbosity):
    agent_root = root_dir / f"agents/TD/{agent_name}"
    if (agent_root / f"{agent_name}.pkl").is_file():
        agent = read_from_file(agent_root / f"{agent_name}.pkl")
        print(f"Loaded agent {agent_name}")
    game = SnakeGame(w, h)

    plot_scores = []
    plot_mean_scores = []
    for k in range(1, n_episodes + 1):
        game.reset()
        state = agent.get_state(game)
        action = agent.get_action(state)
        done = False
        while not done:
            reward, done = game.play_step(action, verbosity>=2)
            next_state = agent.get_state(game)
            next_action = agent.get_action(state)
            # train step
            target = reward + agent.gamma * agent.Q[next_state][next_action][0]
            delta = target - agent.Q[state][action][0]
            agent.Q[state][action][0] += agent.lr * delta
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
        print(f"Saved agent {agent_name}")
    game.quit()
