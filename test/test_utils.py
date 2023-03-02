from algorithms import NetPolicy
import tqdm
import random
import numpy as np
import torch


def set_seed(seed=0):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_game(env, agent: NetPolicy, reward_func=None, render=False):
    state, info= env.reset()
    step = 0
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _, info = env.step(action)
        if step>200:
            done=True
        if reward_func:
            params = {
                "step": step,
                "state": state,
                "next_state": next_state,
                "reward": reward,
                "done": done,
                "info": info,
            }
            reward = reward_func(params)
        agent.store(state, action, reward, done, next_state)
        state = next_state
        step += 1
        total_reward += reward
    return step, total_reward


def train_alog(env, agent: NetPolicy, epi_num, epoch_per_epi, train_start_epi=0,
               reward_func=None, render=False):
    train_bar = tqdm.tqdm(range(epi_num))

    for epi in train_bar:
        step, total_reward = run_game(env, agent, reward_func, render)
        loss = []
        if epi >= train_start_epi:
            for epo in range(epoch_per_epi):
                l = agent.update()
                if l:
                    loss.append(l)
            if loss:
                loss = sum(loss)/len(loss)
                agent.writer.add_scalar(f"train/loss", loss, epi)
        agent.writer.add_scalar(f"train/reward", total_reward, epi)
        agent.writer.add_scalar(f"train/step", step, epi)
        
        train_bar.set_description("r: %.2f"%total_reward)


def eval_alog(env, agent: NetPolicy, epi_num, reward_func=None, render=True):
    for epi in range(epi_num):
        step, total_reward = run_game(env, agent, reward_func, render)
        agent.writer.add_scalar(f"eval/reward", total_reward, epi)
        agent.writer.add_scalar(f"eval/step", step, epi)


def train_eval_algo(env, agent: NetPolicy, train_num, eval_num, train_epoch_per_epi,
                    train_start_epi=0, reward_func=None):
    train_alog(env, agent, train_num,
               train_epoch_per_epi, train_start_epi, reward_func)
    eval_alog(env, agent, eval_num, reward_func)
