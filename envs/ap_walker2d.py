import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class APWalker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, c):
        self.initialized = False
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

        self.c = np.array(c)
        assert len(self.c) == np.prod(self.action_space.shape)
        self.initialized = True

    def step(self, a):
        if self.initialized:
            action_mask = (self.timestep % self.c) == 0
            a = action_mask * a + (1 - action_mask) * self.prev_action

        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()

        if self.initialized:
            self.prev_action = np.array(a)
            self.timestep += 1

        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.timestep = 0
        self.prev_action = self.action_space.sample()

        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
