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

from airgym.lib.utils.isaacgym_utils import RLGPUAlgoObserver
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from airgym.utils.helpers import get_args


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

    if args['num_envs'] > 0:
        config['params']['config']['num_actors'] = args['num_envs']
        config['params']['config']['env_config']['num_envs'] = args['num_envs']

    if args['seed'] > 0:
        config['params']['seed'] = args['seed']
        config['params']['config']['env_config']['seed'] = args['seed']

    return config


if __name__ == '__main__':
    args = vars(get_args())

    config_name = 'ppo_' + args['task']+'.yaml'

    print('Loading config: ', config_name)
    with open(config_name, 'r') as stream:
        config = yaml.safe_load(stream)
    
        config = update_config(config, args)

        from airgym.lib.torch_runner import Runner
        runner = Runner(RLGPUAlgoObserver())
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    print("Config from yaml and args:", config)
    runner.run(args)
