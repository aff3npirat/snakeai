from snakeai.base import QAgentBase


# TODO: TDQLearning
class TDSarsa(QAgentBase):

    def __init__(self, params, name, vision, eps_greedy):
        super().__init__(params, name, vision, eps_greedy)

    def train_episode(self, game):
        game.reset()
        done = False
        state = self.get_state(game)
        action = self.get_action(state)
        while not done:
            done, reward = game.play_step(action)
            next_state = self.get_state(game)
            next_action = self.get_action(next_state)
            target = reward + self.params['gamma'] * self.Q[next_state][next_action]
            delta = target - self.Q[state][action]
            self.Q[state][action] += self.params['lr'] * delta
            state = next_state
            action = next_action
        self.params['n_games'] += 1
