import pickle
import torch
import gym
import torch.nn.functional as F
import numpy as np
from AC_use_dependency import ActorCritic
from AC_use_dependency import ValueNet
from AC_use_dependency import PolicyNet

env_name = 'CartPole-v1'
env = gym.make(env_name)
device = torch.device("cuda")

# file = open('ActorCritic_CartPole0.pkl', 'rb')
file = open('/home/erhalight/Documents/bs/Actor_Critic/ActorCritic_CartPole1.pkl', 'rb')
agent = pickle.load(file)
file.close()

env.reset()
for _ in range(1000):
    env.render()
    env.step(agent.actor(torch.tensor(env.state, dtype=torch.float).to(device)).argmax().item())
env.close()