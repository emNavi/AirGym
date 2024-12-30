
configurations = {}

def get_env_info(env):
    result_shapes = {}
    result_shapes['observation_space'] = env.observation_space
    result_shapes['action_space'] = env.action_space
    result_shapes['agents'] = 1
    result_shapes['value_size'] = 1
    if hasattr(env, "get_number_of_agents"):
        result_shapes['agents'] = env.get_number_of_agents()
    '''
    if isinstance(result_shapes['observation_space'], gym.spaces.dict.Dict):
        result_shapes['observation_space'] = observation_space['observations']
    
    if isinstance(result_shapes['observation_space'], dict):
        result_shapes['observation_space'] = observation_space['observations']
        result_shapes['state_space'] = observation_space['states']
    '''
    if hasattr(env, "value_size"):    
        result_shapes['value_size'] = env.value_size
    print(result_shapes)
    return result_shapes

def get_obs_and_action_spaces_from_config(config):
    env_config = config.get('env_config', {})
    env = configurations[config['env_name']]['env_creator'](**env_config)
    result_shapes = get_env_info(env)
    env.close()
    return result_shapes

def register(name, config):
    configurations[name] = config
    