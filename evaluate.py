import multiprocessing
import time

import gym
import numpy as np
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from tqdm import tqdm

from util import precise_env_name


def make_vectorized_env(env_name, n_envs=multiprocessing.cpu_count()):
    def make_env(env_id, seed=0):
        def _init():
            env = gym.make(env_id)
            env.seed(seed)
            return env

        set_global_seeds(seed)
        return _init

    vec_env = SubprocVecEnv([make_env(precise_env_name(env_name), i) for i in range(n_envs)])
    return vec_env

def evaluate_policy_lsac(vec_env, agent, num_episodes=30, deterministic=True, render=False, discrete=False, ap_env=True):
    episode_rewards = []

    c = vec_env.get_attr('c', [0])[0]
    lcm_c = int(np.lcm.reduce(c))

    with tqdm(total=num_episodes, desc="policy_evaluation", ncols=70) as pbar:
        episode_reward = np.zeros(vec_env.num_envs)
        timestep = np.zeros(vec_env.num_envs, dtype=np.int32)
        obs = vec_env.reset()
        if discrete:
            num_agents, num_actions = len(vec_env.get_attr('ts_ids', [0])[0]), vec_env.get_attr('num_green_phases', [0])[0]
            one_hot_helper = np.eye(num_actions)
            list_action_old = np.array([np.concatenate([one_hot_helper[vec_env.action_space.sample()] for _ in range(lcm_c * num_agents)]) for _ in range(vec_env.num_envs)])
        else:
            list_action_old = np.zeros((vec_env.num_envs, np.prod(vec_env.action_space.shape) * lcm_c))

        assert np.all(np.array(vec_env.get_attr('timestep'), dtype=np.int32) == timestep)
        one_hot_timestep = np.eye(lcm_c)[timestep % lcm_c]
        full_obs = np.concatenate([np.reshape(obs, [vec_env.num_envs, -1]), one_hot_timestep, list_action_old], axis=1)

        while len(episode_rewards) < num_episodes:

            action, _ = agent.predict(obs=full_obs, deterministic=deterministic)
            next_obs, reward, done, _ = vec_env.step(action)

            timestep += 1
            episode_reward = episode_reward + reward
            if np.count_nonzero(done) > 0:
                episode_rewards += list(episode_reward[done])
                episode_reward[done] = 0
                timestep[done] = 0
                pbar.update(np.count_nonzero(done))

            if render:
                vec_env.render()
                print("obs={}, action={}, reward={}, next_obs={}".format(obs, action, reward, next_obs))
                time.sleep(0.1)

            assert np.all(np.array(vec_env.get_attr('timestep'), dtype=np.int32) == timestep)
            if discrete:
                one_hot_action = np.reshape(one_hot_helper[action], [vec_env.num_envs, -1])
                list_action_old = np.concatenate([one_hot_action, list_action_old[:, :-(num_agents * num_actions)]], axis=1)
            else:
                list_action_old = np.concatenate([action, list_action_old[:, :-np.prod(vec_env.action_space.shape)]], axis=1)
            one_hot_timestep = np.eye(lcm_c)[(timestep) % lcm_c]
            full_obs = np.concatenate([np.reshape(next_obs, [vec_env.num_envs, -1]), one_hot_timestep, list_action_old], axis=1)
            obs = next_obs

        episode_rewards = np.array(episode_rewards)

    mu = np.mean(episode_rewards)
    ste = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
    env_id = vec_env.get_attr('spec', [0])[0].id
    print("\n%f +- %f (env_id=%s, deterministic=%s, ap_env=%s, c=%s)" % (mu, ste, env_id, deterministic, ap_env, c))

    return np.mean(episode_rewards)


def evaluate_policy(vec_env, agent, num_episodes=30, deterministic=True, render=False, discrete=False):
    episode_rewards = []
    env_id = vec_env.get_attr('spec', [0])[0].id
    ap_env = 'ap' in env_id.lower()  # True / False

    if ap_env:
        c = vec_env.get_attr('c', [0])[0]
    else:
        if discrete:
            c = np.ones(agent.num_agents)
        else:
            c = np.ones(agent.action_dim)

    with tqdm(total=num_episodes, desc="policy_evaluation", ncols=70) as pbar:
        episode_reward = np.zeros(vec_env.num_envs)
        timestep = np.zeros(vec_env.num_envs, dtype=np.int32)
        obs = vec_env.reset()
        if discrete:
            num_agents, num_actions = len(vec_env.get_attr('ts_ids', [0])[0]), vec_env.get_attr('num_green_phases', [0])[0]
            action_old = np.zeros((vec_env.num_envs, num_agents), dtype=np.int32)
        else:
            action_old = np.zeros((vec_env.num_envs, np.prod(vec_env.action_space.shape)))

        while len(episode_rewards) < num_episodes:
            if ap_env:
                assert np.all(np.array(vec_env.get_attr('timestep'), dtype=np.int32) == timestep)
                action, _ = agent.predict(timestep=vec_env.get_attr('timestep'), action_old=action_old, obs=obs, deterministic=deterministic)
            else:
                action, _ = agent.predict(obs=obs, deterministic=deterministic)
            next_obs, reward, done, _ = vec_env.step(action)
            # print(obs)

            action_mask = np.equal(timestep[:, None] % c, 0)
            action_valid = action_mask * action + (1 - action_mask) * action_old

            timestep += 1
            episode_reward = episode_reward + reward
            if np.count_nonzero(done) > 0:
                episode_rewards += list(episode_reward[done])
                episode_reward[done] = 0
                timestep[done] = 0
                pbar.update(np.count_nonzero(done))

            if render:
                vec_env.render()
                print("obs={}, action={}, reward={}, next_obs={}".format(obs, action, reward, next_obs))
                time.sleep(0.1)

            obs = next_obs
            action_old = action_valid

        episode_rewards = np.array(episode_rewards)

    mu = np.mean(episode_rewards)
    ste = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
    print("\n%f +- %f (env_id=%s, deterministic=%s, ap_env=%s, c=%s)" % (mu, ste, env_id, deterministic, ap_env, c))

    return np.mean(episode_rewards)
