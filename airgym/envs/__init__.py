import os
import traceback
from airgym.utils.task_registry import task_registry

TASK_CONFIGS = [
    {
        'name': 'hovering',
        'config_module': 'base.hovering_config',
        'config_class': 'HoveringCfg',
        'task_module': 'base.hovering',
        'task_class': 'Hovering'
    },
    {
        'name': 'customized',
        'config_module': 'base.customized_config',
        'config_class': 'CustomizedCfg',
        'task_module': 'base.customized',
        'task_class': 'Customized'
    },
    {
        'name': 'balloon',
        'config_module': 'task.balloon_config',
        'config_class': 'BalloonCfg',
        'task_module': 'task.balloon',
        'task_class': 'Balloon'
    },
    {
        'name': 'avoid',
        'config_module': 'task.avoid_config',
        'config_class': 'AvoidCfg',
        'task_module': 'task.avoid',
        'task_class': 'Avoid'
    },
    {
        'name': 'tracking',
        'config_module': 'task.tracking_config',
        'config_class': 'TrackingCfg',
        'task_module': 'task.tracking',
        'task_class': 'Tracking'
    },
    {
        'name': 'planning',
        'config_module': 'task.planning_config',
        'config_class': 'PlanningCfg',
        'task_module': 'task.planning',
        'task_class': 'Planning'
    },
    {
        'name': 'depthgen',
        'config_module': 'base.depthgen_config',
        'config_class': 'DepthGenCfg',
        'task_module': 'base.depthgen',
        'task_class': 'DepthGen'
    }
]

def register_tasks():
    for config in TASK_CONFIGS:
        try:
            config_module = __import__(
                f"airgym.envs.{config['config_module']}", 
                fromlist=[config['config_class']]
            )
            config_class = getattr(config_module, config['config_class'])
            
            task_module = __import__(
                f"airgym.envs.{config['task_module']}", 
                fromlist=[config['task_class']]
            )
            task_class = getattr(task_module, config['task_class'])
            
            task_registry.register(
                config['name'], 
                task_class, 
                config_class()
            )
        except ImportError as e:
            print(f"WARNING! Failed to import {config['name']}: {str(e)}")
            print("Ignore if using on real robot inference.")
            traceback.print_exc()

register_tasks()