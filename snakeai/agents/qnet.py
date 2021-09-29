import numpy as np
import random
from torch import nn, optim
from torch.nn import functional
from collections import deque


from snakeai.helper import dict_to_str, write_to_file


class QNet(nn.Module):

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = functional.relu(self.hidden(x))
        x = self.out(x)
        assert len(np.shape(x)) == 1, f"output shape is {np.shape(x)}"
        return x


# TODO: QNetSarsa
# TODO: test if qnet (without memory train) converges for lr=0.001
# TODO: compare training speed: pytorch vs keras
class QNetLearning:

    def __init__(self, params, name, view, eps_greedy):
        params["n_games"] = 0
        self.params = params
        self.Q = QNet(params['in_size'], params['hidden_size'], 4)
        self.name = name
        self.view = view
        self.eps_greedy = eps_greedy
        self.memory = deque(maxlen=100_000)

    def get_action(self, state):
        action_probs = self.eps_greedy(self.Q[state], self.params)
        return random.choices([0, 1, 2, 3], weights=action_probs)[0]

    def get_state(self, game):
        return self.view(game)

    def train_episode(self, game, render):
        game.reset()
        done = False
        state = self.get_state(game)
        while not done:
            action = self.get_action(state)
            done, reward = game.play_step(action, render)
            next_state = self.get_state(game)
            # train on time step
            self._train(state, action, reward, next_state, done)
            self.memory.append((state, action, reward, next_state, done))
            state = next_state
        # train on full memory
        self._train(*list(zip(*self.memory)), batch_size=1000)
        self.params['n_games'] += 1

    def _train(self, states, actions, rewards, next_states, dones, batch_size=1):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states,
                                                           dones):
            if len(np.shape)
            # output of qnet has shape (1, 4)
            pred = self.Q.forward(state)
            target = pred.clone()

    def save(self, save_dir):
        self.qnet_file = save_dir / f"{self.name}_model"
        self.Q.model.save(self.qnet_file)
        self.Q.model = None
        write_to_file(self, save_dir / f"{self.name}.pkl", text=False)
        info = (f"{type(self).__name__}\n"
                f"({self.eps_greedy.__name__}/{self.view.__name__})\n"
                f"{dict_to_str(self.params)}")
        write_to_file(info, save_dir / f"{self.name}.yml", text=True)
        print(f"Saved {self.name} to '{save_dir}'")

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.Q.model = load_model(self.qnet_file)
