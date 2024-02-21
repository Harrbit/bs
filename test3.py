import mujoco
xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
# print(model.ngeom)
# print(model.geom_rgba)
# try:
#   model.geom()
# except KeyError as e:
#    print(e)
# print(model.geom('green_sphere').rgba)
id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'green_sphere')
# print("geom of green obj is %s" % (model.geom_rgba[id, :]))
# print("id is %s" % (id))
# print('id of "green_sphere": ', model.geom('green_sphere').id)
# print('name of geom 1: ', model.geom(1).name)
# print('name of body 0: ', model.body(0).name)
# print([model.geom(i).name for i in range(model.ngeom)])
data = mujoco.MjData(model)
print("qpos:", data.qvel)
mujoco.mj_kinematics(model, data)
print('raw access:\n', data.geom_xpos)
