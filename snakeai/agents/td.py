from collections import defaultdict

from snakeai.helper import write_to_file, dict_to_str, default_value


# TODO: TDQLearning
class TDSarsa:

    def __init__(self, params, name):
        self.params = params
        self.Q = defaultdict(default_value)
        self.name = name

    def train_episode(self, game, get_state, get_action, render):
        game.reset()
        done = False
        state = get_state(game)
        # if state not in self.Q:
        #     self.Q[state] = [0, 0, 0, 0]
        action = get_action(state)
        while not done:
            reward, done = game.play_step(action, render)
            next_state = get_state(game)

            # if state not in self.Q:
            #     self.Q[state] = [0, 0, 0, 0]
            # if next_state not in self.Q:
            #     self.Q[next_state] = [0, 0, 0, 0]

            next_action = get_action(state)
            target = reward + self.params['gamma'] * self.Q[next_state][next_action]
            delta = target - self.Q[state][action]
            self.Q[state][action] += self.params['lr'] * delta
        self.params['n_games'] += 1

    def save(self, agents_dir):
        save_dir = agents_dir / f"TD/{self.name}"
        write_to_file(self, save_dir / f"{self.name}.pkl", text=False)
        write_to_file(dict_to_str(self.params), save_dir / f"{self.name}.yml", text=True)
        print(f"Saved {self.name} to '{save_dir}'")
