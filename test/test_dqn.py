from torch import nn, optim
import gym
import copy
import sys

sys.path.append('.')

from algorithms import DQNAgent, DoubleDQNAgent
from utils.simple_nets import MLP,DuelingNet, DuelingNet2
from test_utils import train_eval_algo, set_seed
from reward import RewardFuncDict


class ValueNet(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size) -> None:
        super(ValueNet,self).__init__()
        self.net=nn.Sequential(
            MLP(in_size,hidden_sizes,out_size),
        )
    
    def forward(self, x):
        return self.net(x)

class DuelingValueNet(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size) -> None:
        super(DuelingValueNet,self).__init__()
        self.net=nn.Sequential(
            DuelingNet(in_size,hidden_sizes,out_size),
        )
    
    def forward(self, x):
        return self.net(x)

if __name__=="__main__":
    ### test dqn
    env_name = "LunarLander-v2"
    env=gym.make(env_name)
    reward_func = RewardFuncDict[env_name]
    num_actions=env.action_space.n
    obs_shape=env.observation_space.shape
    
    value_net=ValueNet(obs_shape[0],[64,64],num_actions)
    dueling_value_net = DuelingValueNet(obs_shape[0],[64,64],num_actions)
    
    # DQN
    set_seed(0)
    net=copy.deepcopy(value_net).to('cuda')
    optimizer=optim.SGD(net.parameters(),lr=1e-3)
    agent=DQNAgent(0,num_actions,net,optimizer,10000)
    train_eval_algo("DQN",env, agent,1000,100,5, reward_func=reward_func)
    
    # Double DQN
    set_seed(0)
    net=copy.deepcopy(value_net)
    optimizer=optim.SGD(net.parameters(),lr=1e-3)
    agent=DoubleDQNAgent(0,num_actions,net,optimizer,10000)
    train_eval_algo("Double DQN",env, agent,1000,100,5, reward_func=reward_func)
    
    # Dueling DQN
    set_seed(0)
    net=copy.deepcopy(dueling_value_net)
    optimizer=optim.SGD(net.parameters(),lr=1e-3)
    agent=DQNAgent(0,num_actions,net,optimizer,10000)
    train_eval_algo("Dueling DQN",env, agent,1000,100,5, reward_func=reward_func)
    
    # Double Dueling DQN
    set_seed(0)
    net=copy.deepcopy(dueling_value_net)
    optimizer=optim.SGD(net.parameters(),lr=1e-3)
    agent=DQNAgent(0,num_actions,net,optimizer,10000)
    train_eval_algo("Double Dueling DQN",env, agent,1000,100,5, reward_func=reward_func)
    