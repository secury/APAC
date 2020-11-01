import glob
import os

import gym
import numpy as np
import tensorflow as tf

from algorithms.APAC import APAC
from algorithms.LSAC import LSAC
from algorithms.SAC import SAC
from evaluate import make_vectorized_env
from util import precise_env_name

np.set_printoptions(precision=3, suppress=True, linewidth=250)


def run(train_env_name, eval_env_name, alg, seed, total_timesteps, alg_params={}):
    assert train_env_name.replace('ap', '') == eval_env_name.replace('ap', ''), f"Train env '{train_env_name}' is not compatible with the eval env '{eval_env_name}'"
    env = gym.make(precise_env_name(train_env_name))

    log_interval = 100000
    checkpoint_interval = 1000000
    assert checkpoint_interval % log_interval == 0

    if alg in ['sac', 'apac', 'lsac']:
        alg_name = f'{alg}_hidden_{alg_params["hidden_dim"]}'
    else:
        raise NotImplementedError(alg)

    # Set result path
    result_dir = f"eval_results/train_{train_env_name}_eval_{eval_env_name}/{alg_name}"
    checkpoint_dir = f"checkpoint/train_{train_env_name}_eval_{eval_env_name}/{alg_name}/seed_{seed}"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    result_filepath = f"{result_dir}/seed_{seed}.npy"
    checkpoint_path = "%s/step_{}.pkl" % checkpoint_dir
    result = None

    # If already finished, skip
    if os.path.exists(result_filepath):
        print(f'Result file already exists: {result_filepath}')
        return np.load(result_filepath, allow_pickle=True)[()]

    if alg == 'sac':
        model = SAC(env, seed=seed, hidden_dim=alg_params['hidden_dim'])
    elif alg == 'apac':
        apenv = gym.make(precise_env_name('ap' + train_env_name))
        model = APAC(env, c=apenv.c, seed=seed, hidden_dim=alg_params['hidden_dim'])
    elif alg == 'lsac':
        apenv = gym.make(precise_env_name('ap' + train_env_name))
        model = LSAC(env, c=apenv.c, seed=seed, hidden_dim=alg_params['hidden_dim'])
    else:
        raise NotImplementedError(alg)

    # Run algorithm and save the result
    print('==============================================')
    print('Run: ', result_filepath)
    vec_env = make_vectorized_env(eval_env_name)  # for policy evaluation
    result = model.learn(vec_env, total_timesteps=total_timesteps, log_interval=log_interval, checkpoint_interval=checkpoint_interval, seed=seed, result_filepath=result_filepath, checkpoint_path=checkpoint_path, result=result)
    np.save(result_filepath, result)
    os.remove(result_filepath + '.tmp.npy')

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_env_name", help="name of the env to train", default='apwalker_421421')
    parser.add_argument("--eval_env_name", help="name of the env to evaluate", default='apwalker_421421')
    parser.add_argument("--alg", help="name of the algorithm to train (apac / sac)", default='apac')
    parser.add_argument("--total_timesteps", help="total timesteps", default=10000000, type=int)
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    args = parser.parse_args()

    alg_params = {'hidden_dim': 100}

    run(args.train_env_name, args.eval_env_name, args.alg, seed=args.seed, total_timesteps=args.total_timesteps, alg_params=alg_params)
