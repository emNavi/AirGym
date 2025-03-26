import numpy as np
import os
from datetime import datetime
import time
import isaacgym
from airgym.envs import *
from airgym.utils import get_args, task_registry
import torch

def sample_command(args):

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print("Number of environments", env_cfg.env.num_envs)
    command_actions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions))
    
    command_actions[:, 0] = 0.
    command_actions[:, 1] = 0.01
    command_actions[:, 2] = 0.
    command_actions[:, 3] = -0.69
    
    # sp roll +10 yaw 25
    # 
    # w,x,y,z
    # command_actions[:, 0] =  0.973
    # command_actions[:, 1] =  0.085
    # command_actions[:, 2] = -0.019
    # command_actions[:, 3] =  0.216

    # command_actions[:, 0] =  0.973 
    # command_actions[:, 1] =  0.085
    # command_actions[:, 2] =  0.019
    # command_actions[:, 3] =  0.213
    # command_actions[:, 4] = -0.69
    
    # -4 0 0.7 -1.53

    env.reset()
    for i in range(0, 50000):
        obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)
            
        # print("Done", i)
        # if i % 500 == 0:
        #     print("Resetting environment")
        #     print("Shape of observation tensor", obs.shape)
        #     print("Shape of reward tensor", rewards.shape)
        #     print("Shape of reset tensor", resets.shape)
        #     if priviliged_obs is None:
        #         print("Privileged observation is None")
        #     else:
        #         print("Shape of privileged observation tensor", priviliged_obs.shape)
        #     print("------------------")
        #     env.reset()

if __name__ == '__main__':
    args = get_args()
    sample_command(args)
