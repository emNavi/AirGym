from .base_config import BaseConfig
from airgym.utils import asset_register

import numpy as np
import torch
from airgym import AIRGYM_ROOT_DIR

class X152bPx4Cfg(BaseConfig):
    seed = 8
    controller_test = False
    use_tcn = False # if use TCN
    tcn_seqs_len = 25 # if use TCN

    class env:
        target_state = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
        num_envs = 256
        num_observations = 22 + 12
        get_privileged_obs = True # if True the states of all entitites in the environment will be returned as privileged observations, otherwise None will be returned
        env_spacing = 1
        episode_length_s = 24 # episode length in seconds
        num_control_steps_per_env_step = 1 # number of physics steps per env step
        reset_on_collision = False # reset environment when contact force on quadrotor is above a threshold
        create_ground_plane = False # create a ground plane


    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [-5, -5, 4]  # [m]
        lookat = [0, 0, 0]  # [m]

    class sim:
        dt =  0.01
        substeps = 1
        gravity = [0., 0., -9.81] #[0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 0 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


    class asset_config:
        """
        Assets CFG.
        This class defines the assets that will be used in the environment.
        Two types of assets can be defined and managed: a type of assets and the specific asset.
        A Type of Assets: 
            You can choose to include or exclude a type of asset in the environment. All kinds of assets that belong 
            to this type can be randomly selected and and placed in the environment. The number of assets (1~n) to be placed
            can be edit in the asset_params class.

        The Specific Asset:
            You can choose to include or exclude a specific asset in the environment. Only the specified asset will be 
            placed in the environment. Note that the specific asset will not override the type of asset if both are included.
            You must denote the specific position and euler angle of the specific asset!

        If you selet assets, the type of this asset must be included in the asset_type_to_dict_map dictionary, and the class
        of this type must be defined in this class. We have registered some common types of assets in the asset_register.py and
        you can simply inherit from them. If you want to add a new type of asset, you can define a new class in the asset_register.py.
        """
        # assets definitions
        class X152b(asset_register.X152b):
            num_assets = 1