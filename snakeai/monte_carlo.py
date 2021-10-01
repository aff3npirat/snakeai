from collections import defaultdict

from snakeai.base import QAgentBase


class FirstVisit(QAgentBase):

    def __init__(self, params, name, vision, eps_greedy):
        super().__init__(params, name, vision, eps_greedy)

        def default_value():
            return [0, 0, 0, 0]
        self.num_visits = defaultdict(default_value)

    def train_episode(self, game):
        episode = []
        done = False
        while not done:
            state = self.get_state(game)
            action = self.get_action(state)
            done, reward = game.play_step(action)
            episode.append((state, action, reward))
        self.params['n_games'] += 1
        total_reward = 0
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            total_reward = self.params['gamma'] * total_reward + reward
            if (state, action) not in [(s, a) for s, a, _ in episode[0:i]]:
                self.num_visits[state][action] += 1
                self.Q[state][action] += ((total_reward - self.Q[state][action])
                                          / self.num_visits[state][action])


class EveryVisit(FirstVisit):

    def train_episode(self, game):
        episode = []
        done = False
        while not done:
            state = self.get_state(game)
            action = self.get_action(state)
            done, reward = game.play_step(action)
            episode.append((state, action, reward))
        self.params['n_games'] += 1
        total_reward = 0
        for s, a, r in reversed(episode):
            total_reward = total_reward * self.params['gamma'] + r
            self.num_visits[s][a] += 1
            self.Q[s][a] += (total_reward - self.Q[s][a]) / self.num_visits[s][a]
