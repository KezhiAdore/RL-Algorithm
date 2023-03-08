import torch
import copy
from torch import nn


def copy_net_with_noisy(net: nn.Module, sigma: float = 0.0):
    copied_net = copy.deepcopy(net)
    params_dict = copied_net.state_dict()
    for name in params_dict:
        params_dict[name] *= (1+sigma * torch.randn(params_dict[name].shape))
    copied_net.load_state_dict(params_dict)
    return copied_net

def discount_cum(rew, gamma):
    rew = copy.deepcopy(rew)
    for i in reversed(range(len(rew)-1)):
        rew[i] = rew[i] + gamma * rew[i+1]
    return rew

if __name__ == "__main__":
    from simple_nets import MLP
    net = MLP(10,15,2)
    cop_net = copy_net_with_noisy(net,1)
    print(net, cop_net)