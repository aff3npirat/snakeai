import numpy as np
import random
import torch
from collections import deque
from torch import nn, optim
from torch.nn import functional


from snakeai.helper import dict_to_str, write_to_file


class QNet(nn.Module):

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = functional.relu(self.hidden(x))
        x = self.out(x)
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
        self.optimizer = optim.Adam(self.Q.parameters(), lr=params['lr'])
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        state = torch.unsqueeze(state, 0)
        action_values = self.Q(state)
        action_probs = self.eps_greedy(action_values.tolist(), self.params)
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
        self._train_long_memory(1000)
        self.params['n_games'] += 1

    def _train_long_memory(self, batch_size):
        if len(self.memory) > batch_size:
            mini_samples = random.sample(self.memory, batch_size)
        else:
            mini_samples = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_samples)
        self._train(states, actions, rewards, next_states, dones)

    def _train(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)

        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            next_states = torch.unsqueeze(next_states, 0)
            dones = (dones, )

        pred = self.Q(states)
        target = pred.clone()
        for i in range(len(dones)):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.params['gamma'] * max(self.Q(
                    next_states))

        self.Q.zero_grad()  # clears gradient buffers
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

    def save(self, save_dir):
        write_to_file(self, save_dir / f"{self.name}.pkl", text=False)
        info = (f"{type(self).__name__}\n"
                f"({self.eps_greedy.__name__}/{self.view.__name__})\n"
                f"{dict_to_str(self.params)}")
        write_to_file(info, save_dir / f"{self.name}.yml", text=True)
        print(f"Saved {self.name} to '{save_dir}'")

    def __setstate__(self, state):
        self.__dict__.update(state)
