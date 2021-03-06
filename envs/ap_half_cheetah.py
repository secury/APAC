import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class APHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, c):
        self.initialized = False
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

        self.c = np.array(c)
        assert len(self.c) == np.prod(self.action_space.shape)
        self.initialized = True

    def step(self, action):
        if self.initialized:
            action_mask = (self.timestep % self.c) == 0
            action = action_mask * action + (1 - action_mask) * self.prev_action

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False

        if self.initialized:
            self.prev_action = np.array(action)
            self.timestep += 1

        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        self.timestep = 0
        self.prev_action = self.action_space.sample()

        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
