import traceback
import os
import yaml

try:
    from isaacgym import gymutil
    print("isaacgym imported successful.")
except ImportError:
    traceback.print_exc()
    print("isaacgym cannot be imported. Trying to import from gym_utils.")
    from airgym.utils.gym_utils import gymutil
    print("gymutil imported successful from gym_utils.")

from lib.utils.isaacgym_utils import RLGPUAlgoObserver

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def get_args():
    custom_parameters = [
        {"name": "--seed", "type": int, "default": 0, "required": False, "help":  "Random seed, if larger than 0 will overwrite the value in yaml config."},
        
        {"name": "--tf", "required": False, "help": "run tensorflow runner", "action": 'store_true'},
        {"name": "--train", "required": False, "help": "train network", "action": 'store_true'},
        {"name": "--play", "required": False, "help": "play(test) network", "action": 'store_true'},
        {"name": "--checkpoint", "type": str, "required": False, "help": "path to checkpoint"},
        {"name": "--num_envs", "type": int, "default": "4096", "help": "Number of environments to create. Overrides config file if provided."},

        {"name": "--sigma", "type": float, "required": False, "help": "sets new sigma value in case if 'fixed_sigma: True' in yaml config"},
        {"name": "--track",  "action": 'store_true', "help": "if toggled, this experiment will be tracked with Weights and Biases"},
        {"name": "--wandb-project-name", "type": str, "default": "rl_games", "help": "the wandb's project name"},
        {"name": "--wandb-entity", "type": str, "default": None, "help": "the entity (team) of wandb's project"},

        {"name": "--task", "type": str, "default": None, "help": "Override task from config file if provided."},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},

        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--ctl_mode", "required": True, "type": str, "help": 'Specify the control mode and the options are: pos, vel, atti, rate, prop'},
        {"name": "--use_tcn", "type": bool, "default": False, "help": "Use TCN network"},
        {"name": "--tcn_seqs_len", "type": int, "default": 25, "help": "TCN sequence length"},
        ]
        
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


def update_config(config, args):
    if args['task'] is not None:
        config['params']['config']['env_name'] = args['task']
    else:
        args['task'] = 'X152b'
    if args['experiment_name'] is not None:
        config['params']['config']['name'] = args['experiment_name']

    config['params']['config']['env_config']['physics_engine'] = args['physics_engine']
    config['params']['config']['env_config']['sim_device'] = args['sim_device']
    config['params']['config']['env_config']['headless'] = args['headless']
    config['params']['config']['env_config']['use_gpu'] = args['use_gpu']
    config['params']['config']['env_config']['subscenes'] = args['subscenes']
    config['params']['config']['env_config']['use_gpu_pipeline'] = args['use_gpu_pipeline']
    config['params']['config']['env_config']['num_threads'] = args['num_threads']
    config['params']['config']['env_config']['ctl_mode'] = args['ctl_mode']
    config['params']['config']['env_config']['use_tcn'] = args['use_tcn']
    config['params']['config']['env_config']['tcn_seqs_len'] = args['tcn_seqs_len']

    if args['num_envs'] > 0:
        config['params']['config']['num_actors'] = args['num_envs']
        config['params']['config']['env_config']['num_envs'] = args['num_envs']

    if args['seed'] > 0:
        config['params']['seed'] = args['seed']
        config['params']['config']['env_config']['seed'] = args['seed']

    return config


if __name__ == '__main__':
    os.makedirs("../nn", exist_ok=True)
    os.makedirs("../runs", exist_ok=True)

    args = vars(get_args())

    config_name = 'ppo_' + args['task']+'.yaml'

    print('Loading config: ', config_name)
    with open(config_name, 'r') as stream:
        config = yaml.safe_load(stream)
    
        config = update_config(config, args)

        from lib.torch_runner import Runner
        runner = Runner(RLGPUAlgoObserver())
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    print("Config from yaml and args:", config)
    runner.run(args)
