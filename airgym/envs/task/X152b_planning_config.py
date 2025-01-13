from airgym.envs.base.base_config import BaseConfig
from airgym.utils import asset_register

import numpy as np
from airgym import AIRGYM_ROOT_DIR

class X152bPlanningConfig(BaseConfig):
    seed = 1
    controller_test = False

    class env:
        target_state = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]) 
        num_envs = 4 # must be a square number
        num_observations = 22 #+ 12
        headless = True
        get_privileged_obs = True # if True the states of all entitites in the environment will be returned as privileged observations, otherwise None will be returned
        env_spacing = 10  # not used with heightfields/trimeshes
        episode_length_s = 12 # episode length in seconds
        num_control_steps_per_env_step = 1 # number of control & physics steps between camera renders
        enable_onboard_cameras = True # enable onboard cameras
        reset_on_collision = True # reset environment when contact force on quadrotor is above a threshold
        create_ground_plane = True # create a ground plane

        cam_channel = 1
        cam_resolution = (212, 120) # (width, hight)
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
        folder_path = f"{AIRGYM_ROOT_DIR}/resources/models/environment_assets"
        
        include_asset_type = {
            "thin": False,
            "trees": False,
            "objects": False, 
            "cubes": False,
            "flags": False,
            "balls": False,
            "cubes_prim": True,
        }
        
        """
        Note: specific assets will be loaded as the sequence. Please make sure the sequence is correct.
        """
        include_specific_asset = {
            "boundaries/front_wall": False, 
            "boundaries/left_wall": False, 
            "boundaries/top_wall": False, 
            "boundaries/back_wall": False,
            "boundaries/right_wall": False, 
            "boundaries/bottom_wall": False, 
            "boundaries/18X18ground": False,
            "cubes/1X1": False,
            "balls/ball": False,
            "boundaries/8X18ground": False,
        }

        env_lower_bound_min = [-4.0, -8.0, 0.0] # lower bound for the environment space
        env_lower_bound_max = [-4.0, -8.0, 0.0] # lower bound for the environment space
        env_upper_bound_min = [4.0, 8.0, 1.0] # upper bound for the environment space
        env_upper_bound_max = [4.0, 8.0, 1.0] # upper bound for the environment space

        # assets definitions
        class X152b(asset_register.X152b):
            num_assets = 1
            collision_mask = 1

        class ball_asset_params(asset_register.ball_asset_params):
            num_assets = 1
        
        class thin_asset_params(asset_register.thin_asset_params):
            num_assets = 10

        class tree_asset_params(asset_register.tree_asset_params):
            num_assets = 3

        class object_asset_params(asset_register.object_asset_params):
            num_assets = 30

        class cube_asset_params(asset_register.cube_asset_params):
            num_assets = 6
            collision_mask = 0

        class flag_asset_params(asset_register.flag_asset_params):
            num_assets = 6

        class ground(asset_register.ground):
            num_assets = 1
            collision_mask = 1
            specified_position = [[-0, -0, 0.05]]
            specified_euler_angle = [[0.0, 0.0, 0.0]]

        class left_wall(asset_register.left_wall):
            num_assets = 1

        class right_wall(asset_register.right_wall):
            num_assets = 1

        class top_wall(asset_register.top_wall):
            num_assets = 1

        class bottom_wall(asset_register.bottom_wall):
            num_assets = 1

        class front_wall(asset_register.front_wall):
            num_assets = 1

        class back_wall(asset_register.back_wall):
            num_assets = 1

        asset_type_to_dict_map = {
            "balls": ball_asset_params,
            "thin": thin_asset_params,
            "trees": tree_asset_params,
            "objects": object_asset_params,
            "cubes": cube_asset_params,
            "flags": flag_asset_params,
            "cubes_prim": cube_asset_params,
            "boundaries/left_wall": left_wall,
            "boundaries/right_wall": right_wall,
            "boundaries/back_wall": back_wall,
            "boundaries/front_wall": front_wall,
            "boundaries/bottom_wall": bottom_wall,
            "boundaries/top_wall": top_wall,
            "boundaries/8X18ground": ground,
            "boundaries/18X18ground": ground,
            "balls/ball": ball_asset_params,
            }
 