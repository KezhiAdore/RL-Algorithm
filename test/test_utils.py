from algorithms import SingleNetPolicy
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

def run_game(env, agent: SingleNetPolicy, reward_func=None, max_step=None):
    state, info= env.reset()
    step = 0
    total_reward = 0
    done = False
    truncated = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _, info = env.step(action)
        if max_step and (step > max_step):
            done = True
            truncated = True
        if reward_func:
            params = {
                "step": step,
                "state": state,
                "next_state": next_state,
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "info": info,
            }
            reward = reward_func(params)
        agent.store(state, action, reward, done, truncated, next_state)
        state = next_state
        step += 1
        total_reward += reward
    return step, total_reward


def train_algo(env, agent: SingleNetPolicy, epi_num, play_num=1, train_num=1, 
               eval_num=3, train_start_epi=0, reward_func=None, max_step=None):
    agent.train_mode()
    
    train_bar = tqdm.tqdm(range(epi_num))
    for epi in train_bar:
        # collect trajectories
        for _ in range(play_num):
            step, total_reward = run_game(env, agent, reward_func, max_step)
        # train model
        loss = []
        if epi >= train_start_epi:
            for _ in range(train_num):
                l = agent.update()
                if l:
                    loss.append(l)
            if loss:
                loss = sum(loss)/len(loss)
                agent.writer.add_scalar(f"train/loss", loss, epi)
        # eval model
        total_step, total_reward = eval_model(env, agent, eval_num, reward_func, max_step)
        agent.writer.add_scalar(f"train/reward", total_reward, epi)
        agent.writer.add_scalar(f"train/step", total_step, epi) 
        train_bar.set_description("r: %.2f"%total_reward)


def eval_model(env, agent: SingleNetPolicy, epi_num, reward_func=None, max_step=None):
    agent.eval_mode()
    record = {
        "step": [],
        "reward": [],
    }
    for _ in range(epi_num):
        step, total_reward = run_game(env, agent, reward_func, max_step)
        record["step"].append(step)
        record["reward"].append(total_reward)
    
    return np.mean(record["step"]), np.mean(record["reward"])
