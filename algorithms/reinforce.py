import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from tianshou.data.batch import Batch

from utils.simple_nets import MLP
from algorithms.policy import NetPolicy


class ReinforceAgent(NetPolicy):
    def __init__(self, 
                 player_id, 
                 num_actions, 
                 network: nn.Module, 
                 optimizer: optim.Optimizer,
                 buffer_size: int,
                 gamma=0.98,
                 ):
        super().__init__(player_id, num_actions, network, optimizer, buffer_size)
        self._gamma=gamma
    
    def action_probabilities(self, state, legal_action_mask=None):
        if legal_action_mask is None:
            legal_action_mask=[1 for _ in range(self._num_actions)]
        
        state=torch.FloatTensor(state)
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
        
    def update(self,episode):
        def train(trajectory:Batch):
            """update network by a trajectory in game

            Args:
                trajectory (list): _description_
            """
            state=torch.FloatTensor(trajectory["obs"]).to(self.device)
            action=torch.LongTensor(trajectory["act"]).to(self.device)
            reward=torch.FloatTensor(trajectory["rew"]).to(self.device)
            
            # computing discount reward
            discount_reward=torch.zeros_like(reward)
            discount_reward[-1]=reward[-1]
            for t in reversed(range(len(reward)-1)):
                discount_reward[t]=reward[t]+self._gamma*discount_reward[t+1]
                
            # multiply coef
            for t in range(len(discount_reward)):
                discount_reward[t]*=self._gamma**t 
            
            # network forward
            action_probs=self._network(state)
            # loss=-F.cross_entropy(action_probs,action)*discount_total_reward
            probs=torch.sum(action_probs*F.one_hot(action,action_probs.shape[-1]),axis=1)
            log_probs=torch.log(probs)
            loss=-torch.sum(log_probs*discount_reward)
            
            # gradient boosting
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            
            return loss.item()
        
        # split trajectory from replay buffer
        trajectories=[]
        trajectory=[]
        for batch in self._buffer:
            trajectory.append(batch)
            if batch["done"]:
                trajectories.append(Batch(trajectory))
                trajectory=[]
        # train neural network
        loss_record=[]  
        for e in range(episode):
            for trajectory in trajectories:
                loss=train(trajectory)
                loss_record.append(loss)
        self.clear_buffer()
        return np.mean(loss_record)
        
    