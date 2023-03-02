import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from tianshou.data.batch import Batch

from algorithms.policy import NetPolicy


class ReinforceAgent(NetPolicy):
    def __init__(self, 
                 player_id, 
                 num_actions, 
                 network: nn.Module, 
                 optimizer: optim.Optimizer,
                 buffer_size: int,
                 gamma=0.98,
                 log_name=""):
        super().__init__(player_id, num_actions, network, optimizer, buffer_size, log_name=log_name)
        self._gamma=gamma
    
    def action_probabilities(self, state, legal_action_mask=None):
        if legal_action_mask is None:
            legal_action_mask=[1 for _ in range(self._num_actions)]
        
        state=torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs=self._network(state).cpu().numpy()
        
        action_probs*=legal_action_mask # legal mask
        
        if np.sum(action_probs) > 0:
            action_probs /= np.sum(action_probs)
        else:
            action_probs = np.array(
                legal_action_mask / np.sum(legal_action_mask))
        return {action: action_probs[action] for action in range(self._num_actions)} 
    
    def store(self, state, action, reward, done, next_state):
        batch=Batch({
            "obs":state,
            "act":action,
            "rew":reward,
            "done":done,
            "obs_next":next_state,
        })
        self._buffer.add(batch)
        
    def update(self):
        batch = self._buffer.sample(0)[0]
        state = torch.FloatTensor(batch['obs']).to(self.device)
        action = torch.LongTensor(batch['act']).view(-1,1).to(self.device)
        reward = torch.FloatTensor(batch['rew']).to(self.device)
        
        # self._optimizer.zero_grad()
        # G = 0
        # for i in reversed(range(len(reward))):
        #     r = reward[i]
        #     s = torch.FloatTensor([batch["obs"][i]]).to(self.device)
        #     a = torch.LongTensor([batch["act"][i]]).view(-1,1).to(self.device)
        #     log_prob = torch.log(self._network(s)).gather(1,a)
        #     G = self._gamma * G + r
        #     loss = -log_prob * G
        #     loss.backward()
        # self._optimizer.step()
        
        self.optimizer.zero_grad()
        # computing discount reward
        discount_reward=torch.zeros_like(reward)
        discount_reward[-1]=reward[-1]
        for t in reversed(range(len(reward)-1)):
            discount_reward[t]=reward[t]+self._gamma*discount_reward[t+1]
        discount_reward = discount_reward.unsqueeze(-1)
        
        # network forward
        action_probs=self._network(state)
        
        loss = torch.log(action_probs.gather(1, action))
        loss = torch.sum(-loss*discount_reward)
        loss.backward()
        self.optimizer.step()
        
        # gradient boosting
        self.clear_buffer()
        return loss.item()
        
    