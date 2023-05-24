from torch import nn, optim
import gym
import copy
import sys

sys.path.append('.')

from algorithms import SARSAAgent
from utils.simple_nets import MLP
from test_utils import train_algo, set_seed
from reward import RewardFuncDict


class ValueNet(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size) -> None:
        super(ValueNet,self).__init__()
        self.net=nn.Sequential(
            MLP(in_size,hidden_sizes,out_size),
        )
    
    def forward(self, x):
        return self.net(x)

if __name__=="__main__":
    ### test dqn
    from setting import *
    
    set_seed(SEED)
    env=gym.make(ENV_NAME)
    reward_func = RewardFuncDict[ENV_NAME]
    num_actions=env.action_space.n
    obs_shape=env.observation_space.shape
    
    value_net=ValueNet(obs_shape[0],[64,64],num_actions)
    
    # SARSA
    for _ in range(EPISODE):
        set_seed(SEED)
        net=copy.deepcopy(value_net).to(DEVICE)
        optimizer=optim.SGD(net.parameters(),lr=1e-3)
        agent=SARSAAgent(0,num_actions,net,optimizer, log_name="SARSA", max_global_gradient_norm=10)
        train_algo(env, agent, EPISODE_NUM, train_num=5, max_step=MAX_STEP, reward_func=reward_func)