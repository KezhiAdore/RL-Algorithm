import copy
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from tianshou.data.batch import Batch

from .policy import SingleNetPolicy


class SARSAAgent(SingleNetPolicy):
    # SARSA 算法一般不单独进行训练，和策略网络共同配合使用
    # 默认采用q值softmax的值作为动作采样概率

    def __init__(
        self,
        player_id,
        num_actions,
        network: nn.Module,
        optimizer: optim.Optimizer,
        update_num: int=1,
        buffer_size: int=100000,
        min_train_size: int=500,
        target_update_interval: int=20,
        gamma=0.98,
        epsilon=0.9,
        epsilon_min=0.01,
        epsilon_decay_step=1000,
        max_global_gradient_norm: float = None, 
        log_name: str = "",
    ):
        super().__init__(player_id, num_actions, network, optimizer,
                         update_num, gamma, buffer_size, max_global_gradient_norm,
                         log_name)

        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_decay = (epsilon - epsilon_min) / epsilon_decay_step
        self._epsilon_min = epsilon_min
        
        self._min_train_size = min_train_size
        self._target_network = copy.deepcopy(self._network)
        self._target_update_interval = target_update_interval
        
        # last store batch
        self._last_batch = None
        self._update_count = 0
    
    def action_probabilities(self, state, legal_action_mask=None):
        if legal_action_mask is None:
            legal_action_mask = [1 for _ in range(self._num_actions)]
        state = torch.FloatTensor(state)
        with torch.no_grad():
            q_value = F.softmax(self._network(state)).cpu().numpy()
            q_value *= legal_action_mask
            action_probs=q_value/np.sum(q_value)
        return {action: action_probs[action] for action in range(self._num_actions)} 
        
            
    def choose_action(self, state, legal_action_mask=None):
        
        if self._train:
            if np.random.random() < self._epsilon:
                action_probs = self.equal_probabilities(
                    state, legal_action_mask)
                return np.random.choice(list(action_probs.keys()),
                                        p=list(action_probs.values()))

        action_probs = self.action_probabilities(state, legal_action_mask)
        actions=list(action_probs.keys())
        probs=np.array(list(action_probs.values()))
        action=np.random.choice(actions,p=probs)
        return action

    def store(self, state, action, reward, done, next_state):
        if self._last_batch is None:
            self._last_batch={
                "obs": state,
                "act": action,
                "rew": reward,
                "terminated": done,
                "obs_next": next_state,
                "truncated": done,
                "info": {}
            }
        else:
            batch=self._last_batch
            batch["info"]["act_next"]=action
            batch = Batch(batch)
            self.buffer.add(batch)
            self._last_batch={
                "obs": state,
                "act": action,
                "rew": reward,
                "terminated": done,
                "obs_next": next_state,
                "truncated": done,
                "info": {}
            }

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
        next_action = torch.LongTensor(batch["info"]["act_next"]).to(self.device).unsqueeze(-1)
        done = torch.FloatTensor(batch["terminated"]).to(self.device).unsqueeze(-1)

        q_value = self.network(state).gather(1, action)

        next_q_value = self._target_network(next_state).gather(-1,next_action)
        q_target = reward + self._gamma * next_q_value

        loss = F.mse_loss(q_value, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        # update target network
        self._update_count += 1
        if self._update_count % self._target_update_interval == 0:
            self._target_network.load_state_dict(self._network.state_dict())
        
        return loss.item()