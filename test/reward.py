from collections import defaultdict


def cart_pole_v1(params):
    reward = params["reward"]
    step = params["step"]
    done = params["done"]
    if done:
        return -100
    else:
        return reward


def lunar_lander_v2(params):
    reward = params["reward"]
    step = params["step"]
    return reward - step * 0.05


RewardFuncDict = defaultdict(lambda: None)
RewardFuncDict["CartPole-v1"] = cart_pole_v1
RewardFuncDict["CartPole-v0"] = cart_pole_v1
RewardFuncDict["LunarLander-v2"] = lunar_lander_v2
