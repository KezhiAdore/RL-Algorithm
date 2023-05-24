import sys
sys.path.append('.')

from reward import RewardFuncDict
from test_utils import train_algo, set_seed
from utils.simple_nets import MLP
from algorithms import PPOAgent
from torch import nn, optim
from torch.nn import functional as F
import gym
import copy


class BaselineNet(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size) -> None:
        super(BaselineNet, self).__init__()
        self.net = nn.Sequential(
            MLP(in_size, hidden_sizes, out_size),
        )

    def forward(self, x):
        return self.net(x)


class PolicyNet(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size) -> None:
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            MLP(in_size, hidden_sizes, out_size),
        )

    def forward(self, x):
        x = self.net(x)
        x = F.softmax(x, dim=-1)
        return x


if __name__ == "__main__":
    from setting import *
    
    set_seed(SEED)
    env = gym.make(ENV_NAME)
    reward_func = RewardFuncDict[ENV_NAME]
    num_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    val_net = BaselineNet(obs_shape[0], [64, 64], 1)
    policy_net = PolicyNet(obs_shape[0], [64, 64], num_actions)
    
    for _ in range(EPISODE):
        set_seed(SEED)
        pi_net = copy.deepcopy(policy_net).to(DEVICE)
        val_net = copy.deepcopy(val_net).to(DEVICE)
        pi_optimizer = optim.Adam(pi_net.parameters(), lr=1e-3)
        val_optimizer = optim.Adam(val_net.parameters(), lr=1e-3)
        # clipping ppo
        agent = PPOAgent(0, num_actions, pi_net, val_net,pi_optimizer, val_optimizer, algo_type='clip', log_name='clip')
        train_algo(env, agent, EPISODE_NUM, max_step=MAX_STEP, reward_func=reward_func)
    
    for _ in range(EPISODE):
        set_seed(SEED)
        pi_net = copy.deepcopy(policy_net).to(DEVICE)
        val_net = copy.deepcopy(val_net).to(DEVICE)
        pi_optimizer = optim.Adam(pi_net.parameters(), lr=1e-3)
        val_optimizer = optim.Adam(val_net.parameters(), lr=1e-3)
        # none ppo
        agent = PPOAgent(0, num_actions, pi_net, val_net,pi_optimizer, val_optimizer, algo_type='none', log_name='none')
        train_algo(env, agent, EPISODE_NUM, max_step=MAX_STEP, reward_func=reward_func)
    
    for _ in range(EPISODE):
        set_seed(SEED)
        pi_net = copy.deepcopy(policy_net).to(DEVICE)
        val_net = copy.deepcopy(val_net).to(DEVICE)
        pi_optimizer = optim.Adam(pi_net.parameters(), lr=1e-3)
        val_optimizer = optim.Adam(val_net.parameters(), lr=1e-3)
        # adapt ppo
        agent = PPOAgent(0, num_actions, pi_net, val_net,pi_optimizer, val_optimizer, algo_type='adapt', log_name='adapt')
        train_algo(env, agent, EPISODE_NUM, max_step=MAX_STEP, reward_func=reward_func)
    
    for _ in range(EPISODE):
        set_seed(SEED)
        pi_net = copy.deepcopy(policy_net).to(DEVICE)
        val_net = copy.deepcopy(val_net).to(DEVICE)
        pi_optimizer = optim.Adam(pi_net.parameters(), lr=1e-3)
        val_optimizer = optim.Adam(val_net.parameters(), lr=1e-3)
        # adapt ppo
        agent = PPOAgent(0, num_actions, pi_net, val_net,pi_optimizer, val_optimizer, algo_type='all', log_name='all')
        train_algo(env, agent, EPISODE_NUM, max_step=MAX_STEP, reward_func=reward_func)
