from collections import defaultdict

def cart_pole_v1(params):
    reward = params["reward"]
    step = params["step"]
    done = params["done"]
    if done:
        return -100
    else:
        return reward

RewardFuncDict = defaultdict(lambda: None)
RewardFuncDict["CartPole-v1"] = cart_pole_v1
RewardFuncDict["CartPole-v0"] = cart_pole_v1