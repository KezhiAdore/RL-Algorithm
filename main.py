from datetime import datetime
import numpy as np
import gym
from gym import Env
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter

from utils.simple_nets import MLP
from algorithms.reinforce import ReinforceAgent


env=gym.make("CartPole-v1",new_step_api=True)
num_actions=env.action_space.n
obs_shape=env.observation_space.shape


net=MLP(obs_shape[0],[16,64,16],num_actions)

optimizer=optim.SGD(net.parameters(),lr=0.001)
agent=ReinforceAgent(0,num_actions,net,optimizer,100)

    
def run(env:Env,policy,render=True):
    state=env.reset()
    step=0
    done=False
    
    while not done:
        if render:
            env.render()
        action=policy.choose_action(state)
        next_state,reward,done,_,info=env.step(action)
        policy.store(state,action,reward,done,next_state)
        state=next_state
        step+=1
    return step


for i in range(1000):        
    step=run(env,agent)
    print(step)
    loss=agent.update(1)
    agent.writer.add_scalar("loss",loss,i)
    agent.writer.add_scalar("score",step,i)
        