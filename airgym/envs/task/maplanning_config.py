from airgym.envs.base.base_config import BaseConfig
from airgym.assets import *

import numpy as np
from airgym import AIRGYM_ROOT_DIR

class AutoAgentsMeta(type):
    def __new__(cls, name, bases, namespace):
        # 在类创建时设置env.agents
        if 'asset_config' in namespace and 'env' in namespace:
            num_assets = namespace['asset_config'].include_robot["X152b"]["num_assets"]
            namespace['env'].agents = num_assets
            namespace['env'].num_observations = 16 + num_assets*2
            # namespace['env'].num_observations = 16
        return super().__new__(cls, name, bases, namespace)
    
class MAPlanningCfg(BaseConfig, metaclass=AutoAgentsMeta):
    seed = -1
    
    class env:
        agents = None # This will be set dynamically based on the number of assets in asset_config
        target_state = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
        num_envs = 4 # must be a square number
        num_observations = 16
        headless = True
        get_privileged_obs = True # if True the states of all entitites in the environment will be returned as privileged observations, otherwise None will be returned
        env_spacing = 14  # not used with heightfields/trimeshes
        episode_length_s = 16 # episode length in seconds
        num_control_steps_per_env_step = 1 # number of control & physics steps between camera renders
        reset_on_collision = False # reset environment when contact force on quadrotor is above a threshold
        create_ground_plane = True # create a ground plane

        cam_dt = 0.04 # camera render time interval

    class viewer:
        ref_env = 0
        pos = [-5, -5, 4]  # [m]
        lookat = [0, 0, 0]  # [m]

    class sim:
        dt =  0.01 #0.01
        substeps = 1
        gravity = [0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 1 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class asset_config:
        include_robot = {
            "X152b": {
                "num_assets": 4,
                "enable_onboard_cameras": True,
                'cam_channel': 1,
                "enable_tensors": True,
                "width": 212,
                "height": 120,
                "far_plane": 5.0,
                "horizontal_fov": 87.0,
                "use_collision_geometry": True,
                "local_transform.p": (0.15, 0.00, 0.1),
                "local_transform.r": (0.0, 0.0, 0.0, 1.0),
                "collision_mask": 1,
            }
        }

        include_single_asset = {
            "balls/ball": {
                "color": [255,102,102],
                "num_assets": 1,
            },
        }
            
        include_group_asset = {
            "thin": {
                "num_assets": 0, #40
                "collision_mask": 1,
                "color": [139, 69, 0],
            },
        }

        include_boundary = {}