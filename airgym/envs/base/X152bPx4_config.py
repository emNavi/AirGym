from .base_config import BaseConfig

import numpy as np
import torch
from airgym import AIRGYM_ROOT_DIR

class X152bPx4Cfg(BaseConfig):
    seed = 1
    controller_test = False
    class env:
        ctl_mode = "pos"
        target_state = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
        num_envs = 256
        num_observations = 18
        get_privileged_obs = False # if True the states of all entitites in the environment will be returned as privileged observations, otherwise None will be returned
        num_actions = 4 # 9 for attitude
        env_spacing = 1
        episode_length_s = 24 # episode length in seconds
        num_control_steps_per_env_step = 1 # number of physics steps per env step

    class robot_asset:
        file = "{AIRGYM_ROOT_DIR}/resources/robots/X152b/model.urdf"
        name = "X152b"  # actor name
        base_link_name = "base_link"
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fix the base of the robot
        collision_mask = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        density = -1 #0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 100.
        max_linear_velocity = 100.
        armature = 0.001

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
