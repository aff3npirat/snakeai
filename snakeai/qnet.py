import random
import torch
from collections import deque
from torch import nn, optim

from snakeai.base import QAgentBase


class QLearning(QAgentBase):

    def __init__(self, Qnet, params, name, vision, eps_greedy):
        super().__init__(params, name, vision, eps_greedy)
        self.Q = Qnet
        self.memory = deque(maxlen=100_000)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=params['lr'])
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        action_probs = self.eps_greedy(self.Q(state), self.params)
        return random.choices([0, 1, 2, 3], weights=action_probs)[0]

    def train_episode(self, game):
        done = False
        state = self.get_state(game)
        while not done:
            action = self.get_action(state)
            done, reward = game.play_step(action)
            next_state = self.get_state(game)
            self.memory.append((state, action, reward, next_state, done))
            self._train(state, action, reward, next_state, done)
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


class Sarsa(QAgentBase):

    def __init__(self, qnet, params, name, vision, eps_greedy):
        super().__init__(params, name, vision, eps_greedy)
        self.Q = qnet
        self.memory = deque(maxlen=100_000)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=params['lr'])
        self.criterion = nn.MSELoss()

    def train_episode(self, game):
        done = False
        state = self.get_state(game)
        action = self.get_action(state)
        while not done:
            done, reward = game.play_step(action)
            next_state = self.get_state(game)
            next_action = self.get_action(next_state)
            self.memory.append((state, action, reward, next_state, next_action, done))
            self._train(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
        self._train_long_memory(1000)
        self.params['n_games'] += 1

    def _train_long_memory(self, batch_size):
        if len(self.memory) > batch_size:
            mini_sample = random.sample(self.memory, batch_size)
        else:
            mini_sample = self.memory
        self._train(*zip(*mini_sample))

    def _train(self, states, actions, rewards, next_states, next_actions, dones):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.int)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        next_actions = torch.tensor(next_actions, dtype=torch.int)

        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            next_states = torch.unsqueeze(next_states, 0)
            next_actions = torch.unsqueeze(next_actions, 0)
            dones = (dones,)

        pred = self.Q(states)
        target = pred.clone()
        for i in range(len(dones)):
            if dones[i]:
                target[i][actions[i].item()] = rewards[i]
            else:
                gamma = self.params['gamma']
                action = actions[i].item()
                next_action = next_actions[i].item()
                target[i][action] = rewards[i] + gamma * self.Q(next_states[i])[next_action]

        self.Q.zero_grad()  # clear gradient buffers
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
