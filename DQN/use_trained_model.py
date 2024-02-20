import pickle
import torch
import gym
import torch.nn.functional as F
import numpy as np
from DQN_use_dependency import DQN
from DQN_use_dependency import Qnet

env_name = 'CartPole-v1'
env = gym.make(env_name)
device = torch.device("cuda")

# file = open('DQN_CartPole0.pkl', 'rb')
file = open('/home/erhalight/Documents/bs/DQN/DQN_CartPole0.pkl', 'rb')
agent = pickle.load(file)
file.close()

env.reset()
for _ in range(1000):
    env.render()
    env.step(agent.q_net(torch.tensor(env.state, dtype=torch.float).to(device)).argmax().item())
env.close()