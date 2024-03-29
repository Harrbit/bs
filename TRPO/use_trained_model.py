import pickle
import torch
import gym
import torch.nn.functional as F
import numpy as np
from TRPO_use_dependency import TRPO
from TRPO_use_dependency import PolicyNet
from TRPO_use_dependency import ValueNet
from TRPO_use_dependency import TRPOContinuous
from TRPO_use_dependency import PolicyNetContinuous


env_name = 'Humanoid-v3'
# file = open('DQN_CartPole0.pkl', 'rb')
file_path = '/home/erhalight/Documents/bs/TRPO/TRPO_' + env_name + '.pkl'
file = open(file_path, 'rb')
agent = pickle.load(file)
file.close()

for _ in range(100):
    env = gym.make(env_name)
    device = torch.device("cpu")

    state = env.reset()
    done = False
    while not done:
        env.render()
        # next_state, _, done, _ = env.step(agent.actor(torch.tensor(state, dtype=torch.float).to(device)).argmax().item())
        
        # state = torch.tensor([state], dtype=torch.float).to(device)
        # mu, std = agent.actor(state)
        # action_dist = torch.distributions.Normal(mu, std)
        action = agent.take_action(state)
        action = action.cpu().detach().numpy()
        next_state, _, done, _ = env.step(action)

        # next_state, _, done, _ = env.step(agent.actor(torch.tensor(state, dtype=torch.float).to(device)).argmax().item())
        state = next_state
        if done == True:
            break
    env.close()