import pickle
import torch
import gym
import torch.nn.functional as F
import numpy as np
from TRPO_use_dependency import TRPO
from TRPO_use_dependency import PolicyNet
from TRPO_use_dependency import ValueNet

env_name = 'CartPole-v1'
env = gym.make(env_name)
device = torch.device("cpu")

# file = open('DQN_CartPole0.pkl', 'rb')
file = open('/home/erhalight/Documents/bs/TRPO/TRPO_CartPole1.pkl', 'rb')
agent = pickle.load(file)
file.close()

env.reset()
for _ in range(1000):
    env.render()
    env.step(agent.actor(torch.tensor(env.state, dtype=torch.float).to(device)).argmax().item())
env.close()