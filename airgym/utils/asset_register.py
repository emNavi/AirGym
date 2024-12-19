import numpy as np
from airgym import AIRGYM_ROOT_DIR

THIN_SEMANTIC_ID = 1
TREE_SEMANTIC_ID = 2
OBJECT_SEMANTIC_ID = 3
CUBE_SEMANTIC_ID = 4
FLAG_SEMANTIC_ID = 5
BOUNDARY_SEMANTIC_ID = 8

class asset:
    foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
    penalize_contacts_on = []
    terminate_after_contacts_on = []
    disable_gravity = False
    collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
    fix_base_link = False # fix the base of the robot
    collision_mask = 0 # objects with the same collision mask will not collide
    replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
    flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
    density = -1 #0.001
    angular_damping = 0.
    linear_damping = 0.
    max_angular_velocity = 100.
    max_linear_velocity = 100.
    armature = 0.001

class X152b(asset):
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

class asset_state_params(asset):
    num_assets = 1                  # number of assets to include

    min_position_ratio = [0.5, 0.5, 0.5] # min position as a ratio of the bounds
    max_position_ratio = [0.5, 0.5, 0.5] # max position as a ratio of the bounds

    collision_mask = 1

    collapse_fixed_joints = True
    fix_base_link = True
    links_per_asset = 1
    set_whole_body_semantic_mask = False
    set_semantic_mask_per_link = False
    semantic_mask_link_list = [] # For empty list, all links are labeled
    specific_filepath = None # if not None, use this folder instead randomizing
    color = None

class thin_asset_params(asset_state_params):
    num_assets = 10

    collision_mask = 1 # objects with the same collision mask will not collide

    max_position_ratio = [0.95, 0.95, 0.95] # min position as a ratio of the bounds
    min_position_ratio = [0.05, 0.05, 0.05] # max position as a ratio of the bounds

    specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing   the ratios

    min_euler_angles = [-np.pi, -np.pi, -np.pi] # min euler angles
    max_euler_angles = [np.pi, np.pi, np.pi] # max euler angles

    specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
            
    collapse_fixed_joints = True
    links_per_asset = 1
    set_whole_body_semantic_mask = True
    semantic_id = THIN_SEMANTIC_ID
    set_semantic_mask_per_link = False
    semantic_mask_link_list = [] ## If nothing is specified, all links are labeled
    color = [170,66,66]

    # for object convex decomposition
    vhacd_enabled = False
    

class tree_asset_params(asset_state_params):
    num_assets = 1

    collision_mask = 1 # objects with the same collision mask will not collide

    max_position_ratio = [0.95, 0.95, 0.1]
    min_position_ratio = [0.05, 0.05, 0.0]

    specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing   the ratios

    min_euler_angles = [0, -np.pi/6.0, -np.pi] # min euler angles
    max_euler_angles = [0, np.pi/6.0, np.pi] # max euler angles

    specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

    collapse_fixed_joints = True
    links_per_asset = 1
    set_whole_body_semantic_mask = False
    set_semantic_mask_per_link = True
    semantic_mask_link_list = [] ## If nothing is specified, all links are labeled
    semantic_id = TREE_SEMANTIC_ID
    color = [70,200,100]

    # for object convex decomposition
    vhacd_enabled = False

class object_asset_params(asset_state_params):
    num_assets = 30
    
    max_position_ratio = [0.95, 0.95, 0.95] # min position as a ratio of the bounds
    min_position_ratio = [0.05, 0.05, 0.05] # max position as a ratio of the bounds

    specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

    min_euler_angles = [0, -np.pi/6, -np.pi] # min euler angles
    max_euler_angles = [0, np.pi/6, np.pi] # max euler angles

    specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

    links_per_asset = 1
    set_whole_body_semantic_mask = False
    set_semantic_mask_per_link = False
    semantic_id = OBJECT_SEMANTIC_ID

    # for object convex decomposition
    vhacd_enabled = False

class cube_asset_params(asset_state_params):
    num_assets = 20
    
    max_position_ratio = [0.9, 0.9, 0] # min position as a ratio of the bounds
    min_position_ratio = [0.05, 0.05, 0] # max position as a ratio of the bounds

    specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

    min_euler_angles = [0, 0, -np.pi] # min euler angles
    max_euler_angles = [0, 0, np.pi] # max euler angles

    specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

    links_per_asset = 1
    set_whole_body_semantic_mask = True
    set_semantic_mask_per_link = False
    semantic_id = CUBE_SEMANTIC_ID

    # for object convex decomposition
    vhacd_enabled = True
    resolution = 500000

class ball_asset_params(asset_state_params):
    num_assets = 1
    collision_mask = 0 # objects with the same collision mask will not collide
    
    min_position_ratio = [0.5, 0.5, 0.5] # min position as a ratio of the bounds
    max_position_ratio = [0.5, 0.5, 0.5] # max position as a ratio of the bounds

    specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

    min_euler_angles = [0, 0, -np.pi] # min euler angles
    max_euler_angles = [0, 0, np.pi] # max euler angles

    specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

    links_per_asset = 1
    set_whole_body_semantic_mask = True
    set_semantic_mask_per_link = False
    semantic_id = OBJECT_SEMANTIC_ID

    vhacd_enabled = False

class flag_asset_params(asset_state_params):
    num_assets = 6
    
    max_position_ratio = [0.9, 0.9, 0] # min position as a ratio of the bounds
    min_position_ratio = [0.05, 0.05, 0] # max position as a ratio of the bounds

    specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

    min_euler_angles = [0, 0, -np.pi] # min euler angles
    max_euler_angles = [0, 0, np.pi] # max euler angles

    specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

    links_per_asset = 1
    set_whole_body_semantic_mask = True
    set_semantic_mask_per_link = False
    semantic_id = FLAG_SEMANTIC_ID

    # for object convex decomposition
    vhacd_enabled = True
    resolution = 500000

class ground(asset_state_params):
    num_assets = 1

    collision_mask = 1 # objects with the same collision mask will not collide
    
    max_position_ratio = [0.5, .5, 0.05] # min position as a ratio of the bounds
    min_position_ratio = [0.5, .5, 0.05] # max position as a ratio of the bounds

    # specified_position = [-1000, -1000, -1000] # if > -900, use this value instead of randomizing the ratios

    # min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
    # max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

    # specified_euler_angle = [-1000, -1000, -1000] # if > -900, use this value instead of randomizing

    specified_position = [[-0, -0, 0.05]] # if > -900, use this value instead of randomizing the ratios

    min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
    max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

    specified_euler_angle = [[-0, 0, -0]] # if > -900, use this value instead of randomizing

    links_per_asset = 1
    set_whole_body_semantic_mask = True
    set_semantic_mask_per_link = False
    semantic_id = BOUNDARY_SEMANTIC_ID

    # for object convex decomposition
    vhacd_enabled = False

class left_wall(asset_state_params):
    num_assets = 1

    collision_mask = 1 # objects with the same collision mask will not collide
    
    min_position_ratio = [0.5, 1.0, 0.5] # min position as a ratio of the bounds
    max_position_ratio = [0.5, 1.0, 0.5] # max position as a ratio of the bounds

    specified_position = [0.0, 0.0, 0.0] # if > -900, use this value instead of randomizing the ratios

    min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
    max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

    specified_euler_angle = [0.0, 0.0, 0.0] # if > -900, use this value instead of randomizing
            
    collapse_fixed_joints = False
    links_per_asset = 1
    specific_filepath = "cube.urdf"
    semantic_id = BOUNDARY_SEMANTIC_ID
    color = [100,200,210]

    # for object convex decomposition
    vhacd_enabled = False

class right_wall(asset_state_params):
    num_assets = 1

    collision_mask = 1 # objects with the same collision mask will not collide

    min_position_ratio = [0.5, 0.0, 0.5] # min position as a ratio of the bounds
    max_position_ratio = [0.5, 0.0, 0.5] # max position as a ratio of the bounds

    specified_position = [0.0, 0.0, 0.0] # if > -900, use this value instead of randomizing the ratios

    min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
    max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

    specified_euler_angle = [0.0, 0.0, 0.0] # if > -900, use this value instead of randomizing
    
    collapse_fixed_joints = False
    links_per_asset = 1
    specific_filepath = "cube.urdf"
    semantic_id = BOUNDARY_SEMANTIC_ID
    color = [100,200,210]

    # for object convex decomposition
    vhacd_enabled = False

class top_wall(asset_state_params):
    num_assets = 1

    collision_mask = 1 # objects with the same collision mask will not collide

    min_position_ratio = [0.5, 0.5, 1.0] # min position as a ratio of the bounds
    max_position_ratio = [0.5, 0.5, 1.0] # max position as a ratio of the bounds

    specified_position = [0.0, 0.0, 0.0] # if > -900, use this value instead of randomizing the ratios

    min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
    max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

    specified_euler_angle = [0.0, 0.0, 0.0] # if > -900, use this value instead of randomizing

    collapse_fixed_joints = False
    links_per_asset = 1
    specific_filepath = "cube.urdf"
    semantic_id = BOUNDARY_SEMANTIC_ID
    color = [100,200,210]

    vhacd_enabled = False

class bottom_wall(asset_state_params):
    num_assets = 1

    collision_mask = 1 # objects with the same collision mask will not collide

    min_position_ratio = [0.5, 0.5, 0.0] # min position as a ratio of the bounds
    max_position_ratio = [0.5, 0.5, 0.0] # max position as a ratio of the bounds

    specified_position = [0.0, 0.0, 0.0] # if > -900, use this value instead of randomizing the ratios

    min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
    max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

    specified_euler_angle = [0.0, 0.0, 0.0] # if > -900, use this value instead of randomizing

    collapse_fixed_joints = False
    links_per_asset = 1
    semantic_id = BOUNDARY_SEMANTIC_ID
    color = [100,150,150]

    # for object convex decomposition
    vhacd_enabled = False


class front_wall(asset_state_params):
    num_assets = 1

    collision_mask = 1 # objects with the same collision mask will not collide

    min_position_ratio = [1.0, 0.5, 0.5] # min position as a ratio of the bounds
    max_position_ratio = [1.0, 0.5, 0.5] # max position as a ratio of the bounds

    specified_position = [0.0, 0.0, 0.0] # if > -900, use this value instead of randomizing the ratios

    min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
    max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

    specified_euler_angle = [0.0, 0.0, 0.0] # if > -900, use this value instead of randomizing

    collapse_fixed_joints = False
    links_per_asset = 1
    specific_filepath = "cube.urdf"
    semantic_id = BOUNDARY_SEMANTIC_ID
    color = [100,200,210]

    # for object convex decomposition
    vhacd_enabled = False

class back_wall(asset_state_params):
    num_assets = 1

    collision_mask = 1 # objects with the same collision mask will not collide
    
    min_position_ratio = [0.0, 0.5, 0.5] # min position as a ratio of the bounds
    max_position_ratio = [0.0, 0.5, 0.5] # max position as a ratio of the bounds

    specified_position = [0.0, 0.0, 0.0] # if > -900, use this value instead of randomizing the ratios

    min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
    max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

    specified_euler_angle = [0.0, 0.0, 0.0] # if > -900, use this value instead of randomizing
    
    collapse_fixed_joints = False
    links_per_asset = 1
    specific_filepath = "cube.urdf"
    semantic_id = BOUNDARY_SEMANTIC_ID
    color = [100,200,210]

    # for object convex decomposition
    vhacd_enabled = False