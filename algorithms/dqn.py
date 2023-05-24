import copy
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

from .policy import SingleNetPolicy


class DQNAgent(SingleNetPolicy):
    def __init__(self, 
                 player_id, 
                 num_actions: int, 
                 network: nn.Module, 
                 optimizer: optim.Optimizer, 
                 min_train_size=500,
                 target_update_interval=20,
                 gamma: float = 0.98, 
                 epsilon=0.9, 
                 epsilon_min=0.01,
                 epsilon_decay_step=1000,
                 buffer_size: int = 100000, 
                 max_global_gradient_norm: float = None, 
                 log_name: str = "",
                 ):
        super().__init__(player_id, num_actions, network, optimizer, gamma, 
                         buffer_size, max_global_gradient_norm, log_name)

        self._min_train_size = min_train_size
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = (epsilon - epsilon_min) / epsilon_decay_step

        # initial target network
        self._target_network = copy.deepcopy(self._network)
        self._target_update_interval = target_update_interval
        self._update_count = 0
    
    def choose_action(self, state, legal_action_mask=None):
        if self._train:
            if np.random.random() < self._epsilon:
                action_probs = self.equal_probabilities(
                    state, legal_action_mask)
                return np.random.choice(list(action_probs.keys()),
                                        p=list(action_probs.values()))

        if legal_action_mask is None:
            legal_action_mask = [1 for _ in range(self._num_actions)]

        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_value = self.network(state).cpu().numpy()
            q_value *= legal_action_mask
        return np.argmax(q_value)

    def update(self, batch_size=64):
        self._epsilon = np.clip(self._epsilon - self._epsilon_decay, self._epsilon_min, np.inf)
        # check whether the length of replay buffer larger than the min train size
        if len(self.buffer) < self._min_train_size:
            return

        # sample experience from replay buffer
        batch, indexes = self.buffer.sample(batch_size)

        state = torch.FloatTensor(batch["obs"]).to(self.device)
        action = torch.LongTensor(batch["act"]).to(self.device).unsqueeze(-1)
        reward = torch.FloatTensor(batch["rew"]).to(self.device).unsqueeze(-1)
        next_state = torch.FloatTensor(batch["obs_next"]).to(self.device)
        done = torch.FloatTensor(batch["terminated"]).to(self.device).unsqueeze(-1)
        truncated = torch.FloatTensor(batch["truncated"]).to(self.device).unsqueeze(-1)

        q_value = self.network(state).gather(1, action)

        next_q_value = self._target_network(next_state)
        next_action = torch.argmax(next_q_value, dim=1).unsqueeze(-1)
        max_next_q_value = next_q_value.gather(1, next_action)

        q_target = reward + self._gamma * max_next_q_value * (1 - done + truncated)

        loss = F.mse_loss(q_value, q_target)
        self.minimize_with_clipping(self._network,self._optimizer,loss)

        # update target network
        self._update_count += 1
        if self._update_count % self._target_update_interval == 0:
            self._target_network.load_state_dict(self.network.state_dict())

        return loss.item()


class DoubleDQNAgent(DQNAgent):

    def update(self, batch_size=64):
        self._epsilon = np.clip(self._epsilon - self._epsilon_decay, self._epsilon_min, np.inf)
        # check whether the length of replay buffer larger than the min train size
        if len(self.buffer) < self._min_train_size:
            return

        # sample experience from replay buffer
        batch, indexes = self.buffer.sample(batch_size)

        state = torch.FloatTensor(batch["obs"]).to(self.device)
        action = torch.LongTensor(batch["act"]).to(self.device).unsqueeze(-1)
        reward = torch.FloatTensor(batch["rew"]).to(self.device).unsqueeze(-1)
        next_state = torch.FloatTensor(batch["obs_next"]).to(self.device)
        done = torch.FloatTensor(batch["terminated"]).to(self.device).unsqueeze(-1)
        truncated = torch.FloatTensor(batch["truncated"]).to(self.device).unsqueeze(-1)

        q_value = self.network(state).gather(1, action)

        next_q_value = self._target_network(next_state)
        next_action = torch.argmax(self.network(next_state),
                                   dim=1).unsqueeze(-1)
        max_next_q_value = next_q_value.gather(1, next_action)

        q_target = reward + self._gamma * max_next_q_value * (1 - done + truncated)

        loss = F.mse_loss(q_value, q_target)
        self.minimize_with_clipping(self.network, self.optimizer, loss)

        # update target network
        self._update_count += 1
        if self._update_count % self._target_update_interval == 0:
            self._target_network.load_state_dict(self._network.state_dict())

        return loss.item()
