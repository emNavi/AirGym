import numpy as np
import gym
from gym import spaces
from argparse import Namespace

from airgym.envs import *
from airgym.utils import task_registry

from lib.utils import env_configurations
from lib.utils.ivecenv import IVecEnv

vecenv_config = {}

def register(config_name, func):
    vecenv_config[config_name] = func

def create_vec_env(config_name, num_actors, **kwargs):
    vec_env_name = env_configurations.configurations[config_name]['vecenv_type']
    return vecenv_config[vec_env_name](config_name, num_actors, **kwargs)

def get_class_attributes(obj):
    """
    Get attributes from a class or an instance.

    Args:
        obj: The class or instance to extract attributes from.

    Returns:
        A dictionary containing the attribute names and their corresponding values.
    """
    if isinstance(obj, type):
        attributes = {
            key: value
            for key, value in obj.__dict__.items()
            if not key.startswith('__') and not callable(value)
        }
    elif hasattr(obj, '__class__'):
        cls = obj.__class__
        attributes = {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith('__') and not callable(value)
        }
    else:
        raise TypeError("Input must be a class or an instance of a class.")

    return attributes


class ExtractObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)


    def reset(self, **kwargs):
        observations,_privileged_observations = super().reset(**kwargs)
        return observations

    def step(self, action):
        observations, _privileged_observations, rewards, dones, infos = super().step(action)
        
        return (
            observations,
            rewards,
            dones,
            infos,
        )

class AirGymRLGPUEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        print("AirGymRLGPUEnv:", config_name, num_actors, kwargs)
        self.use_image = kwargs.get('use_image', False)
        # print(env_configurations.configurations)
        self.env, self.env_info = env_configurations.configurations[config_name]['env_creator'](**kwargs)

        self.env = ExtractObsWrapper(self.env)

    def step(self, actions):
        return  self.env.step(actions)

    def reset(self):
        return self.env.reset()
    
    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = get_class_attributes(self.env_info.env)
        info['action_space'] = spaces.Box(
            np.ones(self.env.num_actions) * -1., np.ones(self.env.num_actions) * 1.)
        if self.use_image:
            info['observation_space'] = spaces.Dict({
                'image': spaces.Box(0, 1, shape=(self.env.cam_channel, self.env.cam_resolution[0], self.env.cam_resolution[1])),
                'observation': spaces.Box(
                    np.ones(self.env.num_obs) * -np.Inf, np.ones(self.env.num_obs) * np.Inf)
            })
        else:
            info['observation_space'] =  spaces.Box(
                np.ones(self.env.num_obs) * -np.Inf, np.ones(self.env.num_obs) * np.Inf)

        print("Vecenv is reading env config from task_config.py:\n", info)

        return info
    

# auto-register the environment in airgym
for task_name in task_registry.get_registered_tasks(): 
    env_configurations.register(
        task_name,
        {
            'env_creator': lambda **kwargs: task_registry.make_env(task_name, args=Namespace(**kwargs)),
            'vecenv_type': 'AirGym-RLGPU'
        }
    )

# register the vec environment
register('AirGym-RLGPU',
                lambda config_name, num_actors, **kwargs: AirGymRLGPUEnv(config_name, num_actors, **kwargs))