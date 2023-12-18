

import gym
env = gym.make("FrozenLake-v1")
env = env.unwrapped
env.render()

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print("冰洞的索引：", holes)
print("目标的索引：", ends)
for a in env.P[14]:
    print(env.P[14][a])


'''
import gym
from gym.envs.classic_control import rendering


class Test(gym.Env):
    # 如果你不想改参数，下面可以不用写
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.viewer = rendering.Viewer(600, 400)  # 600x400 是画板的长和框

    def render(self, mode='human', close=False):
        # 下面就可以定义你要绘画的元素了
        line1 = rendering.Line((100, 300), (500, 300))
        line2 = rendering.Line((100, 200), (500, 200))
        # 给元素添加颜色
        line1.set_color(0, 0, 0)
        line2.set_color(0, 0, 0)
        # 把图形元素添加到画板中
        self.viewer.add_geom(line1)
        self.viewer.add_geom(line2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


if __name__ == '__main__':
    t = Test()
    while True:
        t.render()

'''
'''
import gym
env = gym.make('MountainCar-v0')
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))
'''
