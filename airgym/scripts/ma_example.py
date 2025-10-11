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
    
    # four actions
    command_actions[:, 0] = 0.
    command_actions[:, 1] = 0.
    command_actions[:, 2] = 1.5
    command_actions[:, 3] = 0

    
    env.reset()
    for i in range(0, 10000000):
        obs, priviliged_obs, rewards, resets, extras = env.step(command_actions.unsqueeze(1).repeat(1, 4, 1))


if __name__ == '__main__':
    args = get_args()
    sample_command(args)
