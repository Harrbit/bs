# import gym  
# import mujoco_py  
# import numpy as np  
# num_joints = 17

# # 创建一个MuJoCo Humanoid环境  
# env = gym.make('Humanoid-v2')  
# env.reset()
# # 查看动作空间  
# action_space = env.action_space  

# # 输出动作空间的范围  
# print(action_space.low)  # 输出每个关节力矩的最小值  
# print(action_space.high) # 输出每个关节力矩的最大值

# # 创建一个符合动作空间要求的动作向量  
# action = np.random.uniform(low=-100, high=100, size=(num_joints,))  
  
# # 将动作向量传递给环境的step方法  
# observation, reward, done, info = env.step(action)

import torch  
  
# 创建一个PyTorch张量  
tensor = torch.tensor([[1, 2], [3, 4]])  
  
# 确保张量在CPU上并且不需要梯度  
tensor = tensor.cpu().detach()  
  
# 将PyTorch张量转换为NumPy数组  
numpy_array = tensor.numpy()  
  
print(numpy_array)