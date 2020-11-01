import pickle
from collections import defaultdict

import numpy as np
from gym.envs.registration import register

C_HOPPER = [[2,1,1], [4,2,1], [6,3,2], [7,5,3]]
C_WALKER = [[2,1,1,2,1,1], [4,2,1,4,2,1], [6,3,2,6,3,2], [7,5,3,7,5,3]]
C_HALFCHEETAH = [[2,1,1,2,1,1], [4,2,1,4,2,1], [6,3,2,6,3,2], [7,5,3,7,5,3]]
C_ANT = [[2,1,2,1,2,1,2,1], [4,2,4,2,4,2,4,2], [6,3,6,3,6,3,6,3], [8,4,8,4,8,4,8,4], [7,5,7,5,7,5,7,5]]

# Stochastic envs
def precise_env_name(problem):
    if problem[:4] == 'apap':
        problem = problem[2:]
    mapping = {
        "ant": "Ant-v2",
        "halfcheetah": "HalfCheetah-v2",
        "hopper": "Hopper-v2",
        "walker": "Walker2d-v2"
    }
    for c in C_HOPPER:
        c_str = ''.join([str(x) for x in c])
        mapping[f"aphopper_{c_str}"] = f"APHopper-{c_str}-v2"
        mapping[f"aphopper_augment_{c_str}"] = f"APHopperAugment-{c_str}-v2"
    for c in C_WALKER:
        c_str = ''.join([str(x) for x in c])
        mapping[f"apwalker_{c_str}"] = f"APWalker2d-{c_str}-v2"
        mapping[f"apwalker_augment_{c_str}"] = f"APWalker2dAugment-{c_str}-v2"
    for c in C_HALFCHEETAH:
        c_str = ''.join([str(x) for x in c])
        mapping[f"aphalfcheetah_{c_str}"] = f"APHalfCheetah-{c_str}-v2"
        mapping[f"aphalfcheetah_augment_{c_str}"] = f"APHalfCheetahAugment-{c_str}-v2"
    for c in C_ANT:
        c_str = ''.join([str(x) for x in c])
        mapping[f"apant_{c_str}"] = f"APAnt-{c_str}-v2"
        mapping[f"apant_augment_{c_str}"] = f"APAntAugment-{c_str}-v2"

    return mapping[problem]


for c in C_HOPPER:
    c_str = ''.join([str(x) for x in c])
    register(
        id=f'APHopper-{c_str}-v2',
        entry_point='envs.ap_hopper:APHopperEnv',
        max_episode_steps=1000,
        reward_threshold=3800.0,
        kwargs={'c': c}
    )
    register(
        id=f'APHopperAugment-{c_str}-v2',
        entry_point='envs.ap_hopper_augment:APHopperAugmentEnv',
        max_episode_steps=1000,
        reward_threshold=3800.0,
        kwargs={'c': c}
    )

for c in C_WALKER:
    c_str = ''.join([str(x) for x in c])
    register(
        id=f'APWalker2d-{c_str}-v2',
        max_episode_steps=1000,
        entry_point='envs.ap_walker2d:APWalker2dEnv',
        kwargs={'c': c}
    )
    register(
        id=f'APWalker2dAugment-{c_str}-v2',
        max_episode_steps=1000,
        entry_point='envs.ap_walker2d_augment:APWalker2dAugmentEnv',
        kwargs={'c': c}
    )

for c in C_HALFCHEETAH:
    c_str = ''.join([str(x) for x in c])
    register(
        id=f'APHalfCheetah-{c_str}-v2',
        entry_point='envs.ap_half_cheetah:APHalfCheetahEnv',
        max_episode_steps=1000,
        kwargs={'c': c}
    )
    register(
        id=f'APHalfCheetahAugment-{c_str}-v2',
        entry_point='envs.ap_half_cheetah_augment:APHalfCheetahAugmentEnv',
        max_episode_steps=1000,
        kwargs={'c': c}
    )

for c in C_ANT:
    c_str = ''.join([str(x) for x in c])
    register(
        id=f'APAnt-{c_str}-v2',
        entry_point='envs.ap_ant:APAntEnv',
        max_episode_steps=1000,
        reward_threshold=6000.0,
        kwargs={'c': c}
    )
    register(
        id=f'APAntAugment-{c_str}-v2',
        entry_point='envs.ap_ant_augment:APAntAugmentEnv',
        max_episode_steps=1000,
        reward_threshold=6000.0,
        kwargs={'c': c}
    )
