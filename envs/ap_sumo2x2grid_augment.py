import gym
import numpy as np
import traci

from envs.sumo_env import SumoEnvironment


class APSumo2x2GridAugmentEnv(SumoEnvironment, gym.core.Env):

    def __init__(self):
        self.initialized = False
        delta_time = 3
        self.c = [2, 3, 3, 2]
        self.lcm_c = np.lcm.reduce(self.c)
        self.timestep = 0
        self.prev_action = np.random.randint(0, 4, 4)
        ts_ids = ('1', '2', '5', '6')
        min_green = {ts_id: delta_time * self.c[i] for i, ts_id in enumerate(ts_ids)}

        self.time_on_phase_to_timestep = {}
        for t in range(self.lcm_c):
            time_on_phase = (t % self.c[0], t % self.c[1], t % self.c[2], t % self.c[3])
            self.time_on_phase_to_timestep[time_on_phase] = t

        super(APSumo2x2GridAugmentEnv, self).__init__(
            net_file='envs/nets/2x2grid/2x2.net.xml',
            route_file='envs/nets/2x2grid/2x2.rou.xml',
            use_gui=False,
            num_seconds=delta_time * 10000,  #100000,
            time_to_load_vehicles=120,
            max_depart_delay=0,
            delta_time=delta_time,
            yellow_time=2,
            min_green=min_green,
            max_green=delta_time * 10,
            phases=[
                traci.trafficlight.Phase(32, "GGrrrrGGrrrr"),
                traci.trafficlight.Phase(2, "yyrrrryyrrrr"),
                traci.trafficlight.Phase(32, "rrGrrrrrGrrr"),
                traci.trafficlight.Phase(2, "rryrrrrryrrr"),
                traci.trafficlight.Phase(32, "rrrGGrrrrGGr"),
                traci.trafficlight.Phase(2, "rrryyrrrryyr"),
                traci.trafficlight.Phase(32, "rrrrrGrrrrrG"),
                traci.trafficlight.Phase(2, "rrrrryrrrrry")
            ])
        self.initialized = True
        self.observation_space.shape = (int(21 + 4 + self.lcm_c),)

    def reset(self):
        obs = super(APSumo2x2GridAugmentEnv, self).reset()
        self.timestep = 0
        self.prev_action = np.random.randint(0, 4, 4)
        return np.concatenate([
            np.tile(np.eye(self.lcm_c)[self.timestep % self.lcm_c], (4, 1)),  # 4 x lcm_c
            np.eye(4)[self.prev_action],  # 4 x 4
            np.array([obs[id] for id in self.ts_ids])
        ], axis=1)

    def step(self, action):
        action_dict = {id: action[i] for i, id in enumerate(self.ts_ids)}
        time_on_phase = tuple(int(self.traffic_signals[id].time_on_phase // self.delta_time) for id in self.ts_ids)
        next_obs, reward, done, info = super(APSumo2x2GridAugmentEnv, self).step(action_dict)

        next_obs = np.concatenate([
            np.tile(np.eye(self.lcm_c)[self.timestep % self.lcm_c], (4, 1)),  # 4 x lcm_c
            np.eye(4)[self.prev_action],  # 4 x 4
            np.array([next_obs[id] for id in self.ts_ids])
        ], axis=1)
        rewards = np.array([reward[id] for id in self.ts_ids])
        # weights = np.exp(-rewards / 0.5); weights /= weights.sum()
        # reward = np.sum(weights * rewards)
        # print('rewards: {} / weights: {} / reward:{}'.format(rewards, weights, reward))
        reward = np.min(rewards)
        done = done['__all__']
        info['timestep'] = self.time_on_phase_to_timestep[time_on_phase]
        info['time_on_phase'] = time_on_phase

        # print('t={}, time_on_phase={}, t={}, a={}, phase={}'.format(
        #     self.sim_step,
        #     info['time_on_phase'],
        #     info['timestep'],
        #     [action_dict[id] for id in self.ts_ids],
        #     [self.traffic_signals[id].phase for id in self.ts_ids]
        # ))

        if self.initialized:
            self.prev_action = np.array(action)
            self.timestep += 1
        return next_obs, reward, done, info
