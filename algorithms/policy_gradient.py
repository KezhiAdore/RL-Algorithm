from algorithms.policy import NetPolicy
from utils import discount_cum
from torch import nn
from torch import optim
from torch.nn import functional as F
from tianshou.data.batch import Batch
import torch
import numpy as np
import collections


class PolicyGradientAgent(NetPolicy):
    def __init__(self,
                 player_id,
                 num_actions: int,
                 network: nn.Module,
                 base_net: nn.Module,
                 optimizer: optim.Optimizer,
                 base_optimizer: optim.Optimizer,
                 buffer_size: int,
                 base_net_update_num: int = 16,
                 gamma: float = 0.98,
                 max_global_gradient_norm: float = None,
                 log_name: str = "",
                 ):
        super().__init__(player_id, num_actions, network,
                         optimizer, buffer_size, max_global_gradient_norm,
                         log_name)

        self._base_net = base_net
        self._base_optimizer = base_optimizer
        self._base_net_update_num = base_net_update_num
        self._gamma = gamma

        self._num_learn = 0
        self._dataset = collections.defaultdict(list)

    def action_probabilities(self, state, legal_action_mask=None):
        if legal_action_mask is None:
            legal_action_mask = [1 for _ in range(self._num_actions)]
        with torch.no_grad():
            act_prob = self._network(torch.FloatTensor(
                state).to(self.device)).cpu().numpy()
        act_prob *= legal_action_mask
        act_prob /= np.sum(act_prob)
        return {act: act_prob[act] for act in range(self._num_actions)}

    def store(self, state, action, reward, done, next_state):
        batch = Batch({
            "obs": state,
            "act": action,
            "rew": reward,
            "done": done,
            "obs_next": next_state,
        })
        self._buffer.add(batch)

    def update(self):
        loss = None
        self._add_buffer_data_to_dataset()
        loss = self._actor_update()
        self._buffer.reset()
        for _ in range(self._base_net_update_num):
            self._value_update()
        
        self._dataset = collections.defaultdict(list)
        return loss

    def _add_buffer_data_to_dataset(self):
        batch = self._buffer.sample(0)[0]
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        done = batch["done"]
        obs_next = batch["obs_next"]
        ret = discount_cum(rew, self._gamma)
        
        adv = np.zeros_like(rew)
        with torch.no_grad():
            obs_ = torch.FloatTensor(obs).to(self.device)
            vals = self._base_net(obs_).cpu().numpy().reshape(-1,)
            adv[:-1] = rew[:-1] - (vals[:-1] - self._gamma * vals[1:])
            adv[-1] = rew[-1]
        adv = discount_cum(adv, self._gamma)
        
        self._dataset["obs"].extend(obs)
        self._dataset["act"].extend(act)
        self._dataset["rew"].extend(rew)
        self._dataset["return"].extend(ret)
        self._dataset["done"].extend(done)
        self._dataset["obs_next"].extend(obs_next)
        self._dataset["adv"].extend(adv)

    def _actor_update(self):
        obs = torch.FloatTensor(self._dataset["obs"]).to(self.device)
        act = torch.LongTensor(self._dataset["act"]).view(-1,1).to(self.device)
        adv = torch.FloatTensor(self._dataset["adv"]).view(-1,1).to(self.device)

        act_prob = self._network(obs)
            
        loss = torch.log(act_prob.gather(1, act))
        loss = torch.sum(-loss * adv)
        self.minimize_with_clipping(self._network, self._optimizer, loss)
        return loss.item()

    def _value_update(self):
        obs = torch.FloatTensor(self._dataset["obs"]).to(self.critic_device)
        ret = torch.FloatTensor(self._dataset["return"]).to(self.critic_device)

        # 计算loss
        value = self._base_net(obs)
        value_loss = F.mse_loss(value, ret.unsqueeze(-1))
        self.minimize_with_clipping(
            self._base_net, self._base_optimizer, value_loss)
        return value_loss.item()

    @property
    def critic_net(self):
        return self._base_net

    @property
    def critic_optimizer(self):
        return self._base_optimizer

    @property
    def critic_device(self):
        return next(self._base_net.parameters()).device
