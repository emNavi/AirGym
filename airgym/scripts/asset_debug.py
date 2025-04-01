
from isaacgym import gymutil
from isaacgym import gymapi

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Body Physics Properties Example")

# configure sim
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.relaxation = 0.9
    sim_params.flex.dynamic_friction = 0.0
    sim_params.flex.static_friction = 0.0
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.static_friction = 0.0
plane_params.dynamic_friction = 0.0

gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 3
spacing = 1.8
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# create list to mantain environment and asset handles
envs = []
actor_handles = []

asset_root = "../../resources/env_assets"
asset_file = "cubes/1X1/model.urdf"
asset_options = gymapi.AssetOptions()
asset_options.disable_gravity = False
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))

def print_all_properties(obj):
    """
    打印类实例中所有 @property 的值。
    
    参数:
    - obj: 类实例。
    """
    # 遍历 dir(obj)，过滤掉以双下划线开头的特殊属性和方法
    properties = [attr for attr in dir(obj) 
                  if not attr.startswith("__") and not callable(getattr(obj, attr))]
    
    print("Properties and their values:")
    for prop in properties:
        try:
            value = getattr(obj, prop)  # 获取属性值
            print(f"{prop}: {value}")
        except Exception as e:
            print(f"{prop}: Error retrieving value ({e})")

print_all_properties(asset_options)
asset_ball = gym.load_asset(sim, asset_root, asset_file, asset_options)

# create static box asset
asset_options.fix_base_link = True

print('Creating %d environments' % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, 1)
    envs.append(env)

    # Scenario 3: Balls with gravity enabled and disabled
    if i == 2:
        # create ball pyramid
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(0, 0, 0, 1)
        num_balls = 3
        ball_radius = .2
        ball_spacing = 2.5 * ball_radius
        y = 1 * (num_balls - 1) * ball_spacing
        while num_balls > 0:
            pose.p = gymapi.Vec3(num_balls * 0.001, 1. + y, 0)
            # create ball actor
            ball_handle = gym.create_actor(env, asset_ball, pose, None)
            color_vec = [1, .2, .2]
            if num_balls != 2:
                color_vec = [.3, .8, .3]
                # Enable gravity back on the middle ball
                body_props = gym.get_actor_rigid_body_properties(env, ball_handle)
                for b in range(len(body_props)):
                    body_props[b].flags = gymapi.RIGID_BODY_NONE
                gym.set_actor_rigid_body_properties(env, ball_handle, body_props)

            # set ball color
            color = gymapi.Vec3(color_vec[0], color_vec[1], color_vec[2])
            gym.set_rigid_body_color(env, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            y += ball_spacing
            num_balls -= 1

# look at the first env
cam_pos = gymapi.Vec3(6, 4.5, 3)
cam_target = gymapi.Vec3(-0.8, 0.5, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

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

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
