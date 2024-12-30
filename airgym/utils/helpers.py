import traceback

try:
    from isaacgym import gymapi
    print("isaacgym imported successful.")
except ImportError:
    traceback.print_exc()
    print("isaacgym cannot be imported. Trying to import from gym_utils.")
    from airgym.utils.gym_utils import gymapi
    print("gymutil imported successful from gym_utils.")


try:
    from isaacgym import gymutil
    print("isaacgym imported successful.")
except ImportError:
    traceback.print_exc()
    print("isaacgym cannot be imported. Trying to import from gym_utils.")
    from airgym.utils.gym_utils import gymutil
    print("gymutil imported successful from gym_utils.")


def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passe1d on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def update_cfg_from_args(env_cfg, args):
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        try:
            env_cfg.env.ctl_mode = args.ctl_mode
        except AttributeError:
            print('ctrl_mode is not exist')
        try:
            env_cfg.seed = args.seed
        except AttributeError:
            print('seed is not exist')
        try:
            env_cfg.controller_test = args.controller_test
        except AttributeError:
            print('controller_test is not exist')
        try:
            env_cfg.use_tcn = args.use_tcn
        except AttributeError:
            print('use_tcn is not exist')
        try:
            env_cfg.tcn_seqs_len = args.tcn_seqs_len
        except AttributeError:
            print('tcn_seqs_len is not exist')
        
        # random seed
                
    return env_cfg

def get_args(additional_parameters=[]):
    custom_parameters = [
        {"name": "--task", "type": str, "default": "X152b", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": None, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--controller_test", "action": "store_true", "default": False, "help": "Test the controller"},
        {"name": "--ctl_mode", "required": True, "type": str, "help": 'Specify the control mode and the options are: pos, vel, atti, rate, prop'},
        {"name": "--use_tcn", "type": bool, "default": False, "help": "Use TCN network"},
        {"name": "--tcn_seqs_len", "type": int, "default": 25, "help": "TCN sequence length"},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters+additional_parameters)

    # name alignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def asset_class_to_AssetOptions(asset_class):
    asset_options = gymapi.AssetOptions()
    asset_options.collapse_fixed_joints = asset_class.collapse_fixed_joints
    asset_options.replace_cylinder_with_capsule = asset_class.replace_cylinder_with_capsule
    asset_options.flip_visual_attachments = asset_class.flip_visual_attachments
    asset_options.fix_base_link = asset_class.fix_base_link
    asset_options.density = asset_class.density
    asset_options.angular_damping = asset_class.angular_damping
    asset_options.linear_damping = asset_class.linear_damping
    asset_options.max_angular_velocity = asset_class.max_angular_velocity
    asset_options.max_linear_velocity = asset_class.max_linear_velocity
    asset_options.disable_gravity = asset_class.disable_gravity
    return asset_options
