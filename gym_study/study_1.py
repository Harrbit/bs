import time
import gym

env = gym.make('BreakoutNoFrameskip-v4')
print(env.observation_space)
print(env.action_space)

obs = env.reset()
for i in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    time.sleep(0.01)
env.close()
print(gym.envs.registry.all())