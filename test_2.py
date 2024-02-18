import os

os.add_dll_directory("C://Users//24716//.mujoco//mujoco200//bin")
os.add_dll_directory("C://Users//24716//.mujoco//mujoco-py//mujoco_py")

import gym
env = gym.make("HalfCheetah-v2", render_mode='human')
# env = gym.wrappers.RecordVideo(env, video_folder='./runs/monitor', name_prefix='mario')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i in range(10):

    obs = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        obs, reward, done, info, _ = env.step(action)
        if done:
           break
env.close()