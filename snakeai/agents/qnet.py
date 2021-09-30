import random
import torch
from collections import deque
from torch import nn, optim
from torch.nn import functional

from snakeai.base import QAgentBase


class QNet(nn.Module):

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = functional.relu(self.hidden(x))
        x = self.out(x)
        return x

    def __getitem__(self, item):
        item = torch.tensor(item, dtype=torch.float)
        if len(item.shape) != 1:
            raise ValueError(f"expected single dimension, got shape {tuple(item.shape)}")
        return self(item).tolist()


# TODO: QNetSarsa
class QNetLearning(QAgentBase):

    def __init__(self, params, name, vision, eps_greedy):
        super().__init__(params, name, vision, eps_greedy)
        self.Q = QNet(params['in_size'], params['hidden_size'], 4)
        self.memory = deque(maxlen=100_000)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=params['lr'])
        self.criterion = nn.MSELoss()

    def train_episode(self, game):
        game.reset()
        done = False
        state = self.get_state(game)
        while not done:
            action = self.get_action(state)
            done, reward = game.play_step(action)
            next_state = self.get_state(game)
            self._train(state, action, reward, next_state, done)
            self.memory.append((state, action, reward, next_state, done))
            state = next_state
        self._train_long_memory(1000)
        self.params['n_games'] += 1

    def _train_long_memory(self, batch_size):
        if len(self.memory) > batch_size:
            mini_sample = random.sample(self.memory, batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self._train(states, actions, rewards, next_states, dones)

    def _train(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.int)
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
                target[i][actions[i].item()] = rewards[i]
            else:
                gamma = self.params['gamma']
                target[i][actions[i].item()] = rewards[i] + gamma * max(self.Q(next_states[i]))

        self.Q.zero_grad()  # clear gradient buffers
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
