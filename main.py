from datetime import datetime
import numpy as np
import gym
import torch
import random
from gym import Env
from torch import optim, nn
from torch.utils.tensorboard.writer import SummaryWriter

from utils.simple_nets import MLP
from algorithms.reinforce import ReinforceAgent
from algorithms.dqn import DQNAgent


class PolicyNet(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size) -> None:
        super(PolicyNet,self).__init__()
        self.net=nn.Sequential(
            MLP(in_size,hidden_sizes,out_size),
            nn.Softmax(),
        )
    
    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size) -> None:
        super(ValueNet,self).__init__()
        self.net=nn.Sequential(
            MLP(in_size,hidden_sizes,out_size),
        )
    
    def forward(self, x):
        return self.net(x)

    
def run(env:Env,policy,render=True):
    state=env.reset()
    step=0
    done=False
    
    while not done:
        if render:
            env.render()
        action=policy.choose_action(state)
        next_state,reward,done,info=env.step(action)
        if done:
            reward=-100
        policy.store(state,action,reward,done,next_state)
        state=next_state
        step+=1
    return step


if __name__=="__main__":
    
    env=gym.make("CartPole-v1")
    num_actions=env.action_space.n
    obs_shape=env.observation_space.shape
    
    net=ValueNet(obs_shape[0],[64,64],num_actions)
    optimizer=optim.SGD(net.parameters(),lr=0.001)

    # agent=ReinforceAgent(0,num_actions,net,optimizer,1000)
    agent=DQNAgent(0,num_actions,net,optimizer,100000)
    
    # setting random seed
    random.seed(0)
    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    for i in range(1000):        
        step=run(env,agent,False)
        print(step)
        agent.update()
        agent.writer.add_scalar("score",step,i)

    run(env,agent)
        