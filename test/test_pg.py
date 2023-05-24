import sys
sys.path.append('.')

from reward import RewardFuncDict
from test_utils import train_algo, set_seed
from utils.simple_nets import MLP
from algorithms import PolicyGradientAgent
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

    b_net = BaselineNet(obs_shape[0], [64, 64], 1)
    policy_net = PolicyNet(obs_shape[0], [64, 64], num_actions)
    
    for _ in range(EPISODE):
        set_seed(SEED)
        actor_net = copy.deepcopy(policy_net).to(DEVICE)
        b_net = copy.deepcopy(b_net).to(DEVICE)
        actor_optimizer = optim.Adam(actor_net.parameters(), lr=1e-3)
        critic_optimizer = optim.Adam(b_net.parameters(), lr=1e-3)
        # policy gradient
        pg_agent = PolicyGradientAgent(0, num_actions, actor_net, b_net,actor_optimizer, critic_optimizer, log_name="pg")
        train_algo(env, pg_agent, EPISODE_NUM, max_step=MAX_STEP)
