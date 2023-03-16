from algorithms import ACNetPolicy
from utils import discount_cum
from torch import nn
from torch import optim
from torch.nn import functional as F
from tianshou.data.batch import Batch
import torch
import numpy as np
import collections


class PolicyGradientAgent(ACNetPolicy):
    def __init__(self, 
                 player_id, 
                 num_actions: int, 
                 pi_net: nn.Module, 
                 v_net: nn.Module, 
                 pi_optimizer: optim.Optimizer, 
                 v_optimizer: optim.Optimizer, 
                 pi_update_num: int = 1, 
                 v_update_num: int = 16, 
                 gamma: float = 0.98, 
                 buffer_size: int = 1000000, 
                 max_global_gradient_norm: float = None, 
                 log_name: str = ""
                 ):
        super().__init__(player_id, num_actions, pi_net, v_net, pi_optimizer, v_optimizer, 
                         pi_update_num, v_update_num, gamma, buffer_size, max_global_gradient_norm, log_name)

        self.dataset = collections.defaultdict(list)

    def action_probabilities(self, state, legal_action_mask=None):
        with torch.no_grad():
            act_prob = self.pi_network(torch.FloatTensor(
                state).to(self.device)).cpu().numpy()
        act_prob = self.legalize_probabilities(act_prob, legal_action_mask)
        return {act: act_prob[act] for act in range(self._num_actions)}

    def store(self, state, action, reward, done, next_state):
        batch = Batch({
            "obs": state,
            "act": action,
            "rew": reward,
            "terminated": done,
            "obs_next": next_state,
            "truncated": done,
        })
        self.buffer.add(batch)

    def update(self):
        loss = None
        self._add_buffer_data_todataset()
        
        for _ in range(self.pi_update_num):
            loss = self._pi_update()
        
        for _ in range(self.val_update_num):
            self._val_update()
        
        self.clear_buffer()
        return loss
    
    def clear_buffer(self):
        super().clear_buffer()
        self.dataset = collections.defaultdict(list)

    def _add_buffer_data_todataset(self):
        batch = self.buffer.sample(0)[0]
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        done = batch["terminated"]
        obs_next = batch["obs_next"]
        ret = discount_cum(rew, self.gamma)
        
        adv = np.zeros_like(rew)
        with torch.no_grad():
            obs_ = torch.FloatTensor(obs).to(self.val_device)
            vals = self.val_network(obs_).cpu().numpy().reshape(-1,)
            adv[:-1] = rew[:-1] - (vals[:-1] - self.gamma * vals[1:])
            adv[-1] = rew[-1]
        adv = discount_cum(adv, self.gamma)
        
        self.dataset["obs"].extend(obs)
        self.dataset["act"].extend(act)
        self.dataset["rew"].extend(rew)
        self.dataset["return"].extend(ret)
        self.dataset["done"].extend(done)
        self.dataset["obs_next"].extend(obs_next)
        self.dataset["adv"].extend(adv)

    def _pi_update(self):
        obs = torch.FloatTensor(np.array(self.dataset["obs"])).to(self.pi_device)
        act = torch.LongTensor(self.dataset["act"]).view(-1,1).to(self.pi_device)
        adv = torch.FloatTensor(self.dataset["adv"]).view(-1,1).to(self.pi_device)

        act_prob = self.pi_network(obs)
            
        loss = torch.log(act_prob.gather(1, act))
        loss = torch.sum(-loss * adv)
        self.minimize_with_clipping(self.pi_network, self.pi_optimizer, loss)
        return loss.item()

    def _val_update(self):
        obs = torch.FloatTensor(self.dataset["obs"]).to(self.val_device)
        ret = torch.FloatTensor(self.dataset["return"]).to(self.val_device)

        # 计算loss
        value = self.val_network(obs)
        value_loss = F.mse_loss(value, ret.unsqueeze(-1))
        self.minimize_with_clipping(
            self.val_network, self.val_optimizer, value_loss)
        return value_loss.item()
