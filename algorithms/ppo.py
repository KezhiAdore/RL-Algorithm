import torch
import numpy as np
import collections
from torch import nn, optim
from torch.nn import functional as F
from tianshou.data.batch import Batch

from .policy import ACNetPolicy
from utils import discount_cum


class PPOAgent(ACNetPolicy):
    """PPO Agent:
        pi_net: policy network, computing an action probability for a given state
        v_net: value network, computing the value of a given state
        
        gamma: discount coefficient
        lam: coefficient of GAE lambda advantage
        
    """
    def __init__(self,
                 player_id,
                 num_actions: int,
                 pi_net: nn.Module,
                 v_net: nn.Module,
                 pi_optimizer: optim.Optimizer,
                 v_optimizer: optim.Optimizer, 
                 pi_update_num: int=16, 
                 v_update_num: int=16, 
                 gamma: float=0.98, 
                 buffer_size: int=1000000, 
                 lam: float=0.98,
                 clip_ratio: float=0.2,
                 target_KL: float=0.01,
                 beta: float=1.5,
                 algo_type: str="clip", # "none", "clip", "adapt", "all"
                 max_global_gradient_norm: float = None, 
                 log_name: str = "",
                 ):
        super().__init__(player_id, num_actions, pi_net, v_net, pi_optimizer, v_optimizer,
                         pi_update_num, v_update_num, gamma, buffer_size, max_global_gradient_norm, log_name)
        
        self._lam = lam
        self._clip_ratio = clip_ratio
        self._target_KL = target_KL
        self._beta = beta
        self._algo_type = algo_type
        
        self.dataset = collections.defaultdict(list)
    
    def action_probabilities(self, state, legal_action_mask=None):
        with torch.no_grad():
            obs = torch.FloatTensor(state).to(self.pi_device)
            act_prob = self.pi_net(obs).cpu().numpy()
        act_prob = self.legalize_probabilities(act_prob, legal_action_mask)
        return {act: act_prob[act] for act in range(self._num_actions)}
    
    def update(self):
        loss = None
        self._buffer_to_dataset()
        for _ in range(self.pi_update_num):
            loss = self._pi_update()
        
        for _ in range(self.val_update_num):
            self._val_update()
            
        self.clear_buffer()
        return loss
    
    def clear_buffer(self):
        super().clear_buffer()
        self.dataset = collections.defaultdict(list)

    def _buffer_to_dataset(self):
        batch = self.buffer.sample(0)[0]
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        done = batch["terminated"]
        obs_next = batch["obs_next"]
        ret = discount_cum(rew, self._gamma)
        
        # GAE-Lambda advantage calculation
        adv = np.zeros_like(rew)
        with torch.no_grad():
            obs_ = torch.FloatTensor(obs).to(self.val_device)
            vals = self.val_net(obs_).cpu().numpy().reshape(-1,)
            adv[:-1] = rew[:-1] - (vals[:-1] - self._gamma * vals[1:])
            adv[-1] = rew[-1]
        adv = discount_cum(adv, self._gamma * self._lam)
        
        # computing log pi
        with torch.no_grad():
            obs_ = torch.FloatTensor(obs).to(self.pi_device)
            act_ = torch.LongTensor(act).to(self.pi_device)
            log_p = torch.log(self.pi_net(obs_).gather(1, act_.view(-1,1)))
        
        self.dataset["obs"].extend(obs)
        self.dataset["act"].extend(act)
        self.dataset["rew"].extend(rew)
        self.dataset["return"].extend(ret)
        self.dataset["done"].extend(done)
        self.dataset["obs_next"].extend(obs_next)
        self.dataset["adv"].extend(adv)
        self.dataset["log_p"].extend(log_p)
    
    def _pi_update(self):
        # no clipping or penalty
        device = self.pi_device
        obs = torch.FloatTensor(self.dataset["obs"]).to(device)
        act = torch.LongTensor(self.dataset["act"]).to(device).unsqueeze(-1)
        adv = torch.FloatTensor(self.dataset["adv"]).to(device).unsqueeze(-1)
        log_p_old = torch.FloatTensor(self.dataset["log_p"]).to(device).unsqueeze(-1)
        
        pi = self.pi_net(obs)
        log_p = torch.log(pi.gather(1, act.view(-1,1)))
        
        ratio = torch.exp(log_p - log_p_old)
        if self._algo_type == "clip" or self._algo_type == "all":
            ratio = torch.clip(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
        loss = torch.mean(ratio * adv)
        
        if self._algo_type == "adapt" or self._algo_type == "all":
            approx_KL = torch.mean(log_p - log_p_old)
            # update loss
            loss += self._beta * approx_KL
            # update beta
            if approx_KL < self._target_KL/1.5:
                self._beta *= 2
            if approx_KL > self._target_KL/1.5:
                self._beta *= 2
        
        self.minimize_with_clipping(self.pi_net, self.pi_optimizer, -loss)
        return loss.item()
         
    
    def _val_update(self):
        obs = torch.FloatTensor(self.dataset["obs"]).to(self.val_device)
        ret = torch.FloatTensor(self.dataset["return"]).to(self.val_device).unsqueeze(-1)
        
        val = self.val_net(obs)
        loss = F.mse_loss(val, ret)
        self.minimize_with_clipping(self.val_net, self.val_optimizer, loss)
        return loss.item()