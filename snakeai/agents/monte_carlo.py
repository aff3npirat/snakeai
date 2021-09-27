from collections import defaultdict

from snakeai.helper import default_value, dict_to_str, write_to_file


# TODO: EveryVisitMC
class FirstVisitMC:

    def __init__(self, params, name):
        self.params = params
        self.name = name
        self.Q = defaultdict(default_value)
        self.num_visits = defaultdict(default_value)

    def train_episode(self, game, get_state, get_action, render):
        game.reset()
        episode = []
        done = False
        while not done:
            state = get_state(game)
            action = get_action(state)
            done, reward = game.play_step(action, render)
            episode.append((state, action, reward))
        self.params['n_games'] += 1
        total_reward = 0
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]

            # if state not in self.Q:
            #     self.Q[state] = [0, 0, 0, 0]
            #     self.num_visits[state] = [0, 0, 0, 0]

            total_reward = self.params['gamma'] * total_reward + reward
            if (state, action) not in [(s, a) for s, a, _ in episode[0:i]]:
                self.num_visits[state][action] += 1
                self.Q[state][action] += ((total_reward - self.Q[state][action])
                                          / self.num_visits[state][action])

    def save(self, agents_dir):
        save_dir = agents_dir / f"monte_carlo/{self.name}"
        write_to_file(self, save_dir / f"{self.name}.pkl", text=False)
        write_to_file(dict_to_str(self.params), save_dir / f"{self.name}.yml", text=True)
        print(f"Saved {self.name} to '{save_dir}'")
