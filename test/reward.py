from collections import defaultdict


def cart_pole_v1(params):
    reward = params["reward"]
    step = params["step"]
    done = params["done"]
    truncated = params["truncated"]
    if done and (not truncated):
        return -100
    else:
        return reward

def lunar_lander_v2(params):
    reward = params["reward"]
    step = params["step"]
    return reward - step * 0.05

def mountain_car_v0(params):
    reward = params["reward"]
    state = params["state"]
    step = params["step"]
    
    x, v = state
    return reward + 1 + max(x+0.5, 0)

RewardFuncDict = defaultdict(lambda: None)
RewardFuncDict["CartPole-v1"] = cart_pole_v1
RewardFuncDict["CartPole-v0"] = cart_pole_v1
RewardFuncDict["LunarLander-v2"] = lunar_lander_v2
RewardFuncDict["MountainCar-v0"] = mountain_car_v0
