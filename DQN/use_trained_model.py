import pickle
import torch
import gym

env_name = 'MountainCar-v0'
env = gym.make(env_name)
device = torch.device("cuda")

f = open('/home/erhalight/Documents/bs/DQN/DQN_CartPole0.pkl', 'rb')
agent = pickle.load(f)
f.close()

env.reset()
for _ in range(10000):
    env.render()
    env.step(agent.q_net(torch.tensor(env.state, dtype=torch.float).to(device)).argmax().item())
env.close()