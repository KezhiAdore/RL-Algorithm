import sys
sys.path.append('.')

from torch.nn import functional as F
from torch import nn, optim
from algorithms import ReinforceAgent
from utils.simple_nets import MLP
from test_utils import train_algo, set_seed
from reward import RewardFuncDict
import gym
import copy


class PolicyNet(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size) -> None:
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            MLP(in_size, hidden_sizes, out_size),
        )

    def forward(self, x):
        x = self.net(x)
        x = F.softmax(x)
        return x


if __name__ == "__main__":
    # test REINFORCE
    from setting import *

    set_seed(SEED)
    env = gym.make(ENV_NAME)
    reward_func = RewardFuncDict[ENV_NAME]
    num_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    policy_net = PolicyNet(obs_shape[0], [128, 128], num_actions)
    # REINFORCE
    for _ in range(EPISODE):
        set_seed(SEED)
        net = copy.deepcopy(policy_net)
        optimizer = optim.Adam(net.parameters(), lr=3e-4)
        agent = ReinforceAgent(0, num_actions, net, optimizer, BUFFER_SIZE)
        train_algo(env, agent, EPISODE_NUM, max_step=MAX_STEP, reward_func=reward_func)
