import mujoco_py
import copy
import numpy as np
model = mujoco_py.load_model_from_path("/home/erhalight/.mujoco/mujoco210/model/humanoid.xml")
sim = mujoco_py.MjSim(model, nsubsteps=1000)
state = copy.deepcopy(sim.get_state())

def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

for name in sim.model.joint_names:
    # sim_state = sim.data.qpos.flat[:]
    qpos = sim.data.get_joint_qpos(name)
    qvel = sim.data.get_joint_qvel(name)
    # print(sim_state)
    print(name,":\n" , "qpos:", qpos ,"qvel:", qvel)

qpos = np.array([0.0, 0.0, 0.454, 1.0, 0.0, 0.0, 0.0,
                               0.0, -0.87, 1.78,
                               0.0, -0.87, 1.78,
                               0.0, -0.87, 1.78,
                               0.0, -0.87, 1.78])

old_state = sim.get_state()
old_state.qvel[15] += 1
print(len(old_state.qpos))
new_state = mujoco_py.MjSimState(old_state.time, old_state.qpos, old_state.qvel, old_state.act, old_state.udd_state)
sim.set_state(new_state)

viewer = mujoco_py.MjViewer(sim)
viewer.render()

