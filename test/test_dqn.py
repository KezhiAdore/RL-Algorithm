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
    from setting import *
    
    set_seed(SEED)
    env=gym.make(ENV_NAME)
    reward_func = RewardFuncDict[ENV_NAME]
    num_actions=env.action_space.n
    obs_shape=env.observation_space.shape
    
    value_net=ValueNet(obs_shape[0],[64,64],num_actions)
    dueling_value_net = DuelingValueNet(obs_shape[0],[64,64],num_actions)
    
    # DQN
    for _ in range(EPISODE):
        set_seed(SEED)
        net=copy.deepcopy(value_net).to(DEVICE)
        optimizer=optim.SGD(net.parameters(),lr=1e-3)
        agent=DQNAgent(0,num_actions,net,optimizer, log_name="DQN")
        train_eval_algo(env, agent,TRAIN_STEP,EVAL_STEP,5, reward_func=reward_func, max_step=MAX_STEP)
    
    # Double DQN
    for _ in range(EPISODE):
        set_seed(SEED)
        net=copy.deepcopy(value_net).to(DEVICE)
        optimizer=optim.SGD(net.parameters(),lr=1e-3)
        agent=DoubleDQNAgent(0,num_actions,net,optimizer, log_name="DoubleDQN")
        train_eval_algo(env, agent,TRAIN_STEP,EVAL_STEP,5, reward_func=reward_func, max_step=MAX_STEP)
    
    # Dueling DQN
    for _ in range(EPISODE):
        set_seed(SEED)
        net=copy.deepcopy(dueling_value_net).to(DEVICE)
        optimizer=optim.SGD(net.parameters(),lr=1e-3)
        agent=DQNAgent(0,num_actions,net,optimizer,log_name="DuelingDQN")
        train_eval_algo(env, agent,TRAIN_STEP,EVAL_STEP,5, reward_func=reward_func, max_step=MAX_STEP)
    
    # Double Dueling DQN
    for _ in range(EPISODE):
        set_seed(SEED)
        net=copy.deepcopy(dueling_value_net).to(DEVICE)
        optimizer=optim.SGD(net.parameters(),lr=1e-3)
        agent=DoubleDQNAgent(0,num_actions,net,optimizer,log_name="DoubleDuelingDQN")
        train_eval_algo(env, agent,TRAIN_STEP,EVAL_STEP,5, reward_func=reward_func, max_step=MAX_STEP)
    