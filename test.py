import torch
import os
print(torch.__version__)
print(torch.cuda.is_available())
os.add_dll_directory("C://Users//24716//.mujoco//mujoco200//bin")
os.add_dll_directory("C://Users//24716//.mujoco//mujoco-py//mujoco_py")
import mujoco_py
mj_path, _ = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid100.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
print(sim.data.qpos)

sim.step()
print(sim.data.qpos)
print("hello world")
