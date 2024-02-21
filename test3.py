# import mujoco
# import mediapy as media
# import matplotlib.pyplot as plt
# import numpy as np
# import copy
# import mujoco_py


# xml = """
# <mujoco>
#   <worldbody>
#     <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
#     <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
#   </worldbody>
# </mujoco>
# """
# # model = mujoco.MjModel.from_xml_string(xml)
# # # print(model.ngeom)
# # # print(model.geom_rgba)
# # # try:
# # #   model.geom()
# # # except KeyError as e:
# # #    print(e)
# # # print(model.geom('green_sphere').rgba)
# # id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'green_sphere')
# # # print("geom of green obj is %s" % (model.geom_rgba[id, :]))
# # # print("id is %s" % (id))
# # # print('id of "green_sphere": ', model.geom('green_sphere').id)
# # # print('name of geom 1: ', model.geom(1).name)
# # # print('name of body 0: ', model.body(0).name)
# # # print([model.geom(i).name for i in range(model.ngeom)])
# # data = mujoco.MjData(model)
# # # print("qpos:", data.qpos)
# # mujoco.mj_kinematics(model, data)
# # # print('raw access:\n', data.geom_xpos)
# # renderer = mujoco.Renderer(model)
# # media.show_image(renderer.render())

# # mujoco.mj_forward(model, data)
# # renderer.update_scene(data)

# # media.show_image(renderer.render())

# model = mujoco_py.load_model_from_path("/home/erhalight/.mujoco/mujoco210/model/humanoid.xml")
# sim = mujoco_py.MjSim(model, nsubsteps=1000)

# state = copy.deepcopy(sim.get_state())

# viewer = mujoco_py.MjViewer(sim)
# # viewer.render()


# data = viewer.read_pixels(100, 100, depth=False)

import mujoco_py
import copy
import numpy as np
model = mujoco_py.load_model_from_path("/home/erhalight/.mujoco/mujoco210/model/humanoid.xml")
sim = mujoco_py.MjSim(model, nsubsteps=1000)
state = copy.deepcopy(sim.get_state())

# for name in sim.model.joint_names:
#     sim_state = sim.data.qpos.flat[:]
#     # qpos = sim.data.get_joint_qpos(name)
#     # qvel = sim.data.get_joint_qvel(name)
#     print(sim_state)
#     # print(name,":\n" , "qpos:", qpos ,"qvel:", qvel)

qpos = np.array([0.0, 0.0, 0.454, 1.0, 0.0, 0.0, 0.0,
                               0.0, -0.87, 1.78,
                               0.0, -0.87, 1.78,
                               0.0, -0.87, 1.78,
                               0.0, -0.87, 1.78])

state.qpos = qpos
# sim.set_state(state)

viewer = mujoco_py.MjViewer(sim)
# viewer.render()

