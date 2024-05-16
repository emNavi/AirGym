
from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch

import numpy as np
import torch

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Example of applying forces and torques to bodies")

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
elif args.physics_engine == gymapi.SIM_FLEX and not args.use_gpu_pipeline:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
else:
    raise Exception("GPU pipeline is only available with PhysX")

sim_params.use_gpu_pipeline = args.use_gpu_pipeline
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# load assets
asset_root = "../../resources/models"
ground_model = "grounds/18X18ground/model.urdf"
ground_asset = gym.load_asset(sim, asset_root, ground_model, gymapi.AssetOptions())

cube1X4_model = "objects/cubes/1X4/model.urdf"
cube1X4_asset = gym.load_asset(sim, asset_root, cube1X4_model, gymapi.AssetOptions())

cube2X2square_model = "objects/cubes/2X2square/model.urdf"
cube2X2square_asset = gym.load_asset(sim, asset_root, cube2X2square_model, gymapi.AssetOptions())

cube2X3_model = "objects/cubes/2X3/model.urdf"
cube2X3_asset = gym.load_asset(sim, asset_root, cube2X3_model, gymapi.AssetOptions())

cube2X4_model = "objects/cubes/2X4/model.urdf"
cube2X4_asset = gym.load_asset(sim, asset_root, cube2X4_model, gymapi.AssetOptions())

cube2X4arch_model = "objects/cubes/2X4arch/model.urdf"
cube2X4arch_asset = gym.load_asset(sim, asset_root, cube2X4arch_model, gymapi.AssetOptions())

cube3X3arch_model = "objects/cubes/3X3arch/model.urdf"
cube3X3arch_asset = gym.load_asset(sim, asset_root, cube3X3arch_model, gymapi.AssetOptions())

cube3X4arch_model = "objects/cubes/3X4arch/model.urdf"
cube3X4arch_asset = gym.load_asset(sim, asset_root, cube3X4arch_model, gymapi.AssetOptions())

arch1_6m_model = "objects/arch1_6m/model.urdf"
arch1_6m_asset = gym.load_asset(sim, asset_root, arch1_6m_model, gymapi.AssetOptions())

circle1_5m_model = "objects/circle1_5m/model.urdf"
circle1_5m_asset = gym.load_asset(sim, asset_root, circle1_5m_model, gymapi.AssetOptions())

circle2_5m_model = "objects/circle2_5m/model.urdf"
circle2_5m_asset = gym.load_asset(sim, asset_root, circle2_5m_model, gymapi.AssetOptions())

circle2m_model = "objects/circle2m/model.urdf"
circle2m_asset = gym.load_asset(sim, asset_root, circle2m_model, gymapi.AssetOptions())


# default pose
pose = gymapi.Transform()
pose.p.z = 0.0

# set up the env grid
num_envs = 1
num_per_row = int(np.sqrt(num_envs))
env_spacing = 5
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# set random seed
np.random.seed(12)

envs = []
handles = []
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    pose.p = gymapi.Vec3(-8, -8, -0.01)
    ground_handle = gym.create_actor(env, ground_asset, pose, "actor", i, 1)
    for i in range(0, 1): 
        pose.p = gymapi.Vec3((np.random.random(1)-1)*16, (np.random.random(1)-1)*18, 0)
        pose.r = gymapi.Quat(0, 0, (np.random.random(1)-1), 1)
        circle2m_handle = gym.create_actor(env, circle2m_asset, pose, "actor", i, 0)
    for i in range(0, 1): 
        pose.p = gymapi.Vec3((np.random.random(1)-1)*16, (np.random.random(1)-1)*16, 0)
        pose.r = gymapi.Quat(0, 0, (np.random.random(1)-1), 1)
        arch1_6m_handle = gym.create_actor(env, arch1_6m_asset, pose, "actor", i, 0)
    for i in range(0, 2):
        pose.p = gymapi.Vec3((np.random.random(1)-1)*16, (np.random.random(1)-1)*16, 0)
        pose.r = gymapi.Quat(0, 0, (np.random.random(1)-1), 1)
        circle1_5m_handle = gym.create_actor(env, circle1_5m_asset, pose, "actor", i, 0)
    for i in range(0, 2):
        pose.p = gymapi.Vec3((np.random.random(1)-1)*16, (np.random.random(1)-1)*16, 0)
        pose.r = gymapi.Quat(0, 0, (np.random.random(1)-1), 1)
        circle2_5m_handle = gym.create_actor(env, circle2_5m_asset, pose, "actor", i, 0)
    for i in range(0, 2):
        pose.p = gymapi.Vec3((np.random.random(1)-1)*16, (np.random.random(1)-1)*16, 0)
        pose.r = gymapi.Quat(0, 0, (np.random.random(1)-1), 1)
        cube3X3arch_handle = gym.create_actor(env, cube3X3arch_asset, pose, "actor", i, 0)
    for i in range(0, 3):
        pose.p = gymapi.Vec3((np.random.random(1)-1)*16, (np.random.random(1)-1)*16, 0)
        pose.r = gymapi.Quat(0, 0, (np.random.random(1)-1), 1)
        cube2X4_handle = gym.create_actor(env, cube2X4_asset, pose, "actor", i, 0)
    for i in range(0, 3):
        pose.p = gymapi.Vec3((np.random.random(1)-1)*16, (np.random.random(1)-1)*16, 0)
        pose.r = gymapi.Quat(0, 0, (np.random.random(1)-1), 1)
        cube2X4arch_handle = gym.create_actor(env, cube2X4arch_asset, pose, "actor", i, 0)
    for i in range(0, 3):
        pose.p = gymapi.Vec3((np.random.random(1)-1)*16, (np.random.random(1)-1)*16, 0)
        pose.r = gymapi.Quat(0, 0, (np.random.random(1)-1), 1)
        cube2X2square_handle = gym.create_actor(env, cube2X2square_asset, pose, "actor", i, 0)
    for i in range(0, 7):
        pose.p = gymapi.Vec3((np.random.random(1)-1)*16, (np.random.random(1)-1)*16, 0)
        pose.r = gymapi.Quat(0, 0, (np.random.random(1)-1), 1)
        cube1X4_handle = gym.create_actor(env, cube1X4_asset, pose, "actor", i, 0)
    for i in range(0, 12):
        pose.p = gymapi.Vec3((np.random.random(1)-1)*16, (np.random.random(1)-1)*16, 0)
        pose.r = gymapi.Quat(0, 0, (np.random.random(1)-1), 1)
        cube2X3_handle = gym.create_actor(env, cube2X3_asset, pose, "actor", i, 0)
    for i in range(0, 6):
        pose.p = gymapi.Vec3((np.random.random(1)-1)*16, (np.random.random(1)-1)*16, 0)
        pose.r = gymapi.Quat(0, 0, (np.random.random(1)-1), 1)
        cube3X4arch_handle = gym.create_actor(env, cube3X4arch_asset, pose, "actor", i, 0)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 20, 5), gymapi.Vec3(0, 0, 1))

gym.prepare_sim(sim)

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
