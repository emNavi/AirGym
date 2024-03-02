from airgym.envs.base.aerial_robot_config import AerialRobotCfg
from airgym.envs.base.aerial_robot_with_obstacles_config import AerialRobotWithObstaclesCfg
from .base.aerial_robot  import AerialRobot
from .base.aerial_robot_with_obstacles import AerialRobotWithObstacles

from .base.X152bPx4_config import X152bPx4Cfg
from .base.X152bPx4 import X152bPx4

import os

from airgym.utils.task_registry import task_registry

task_registry.register( "quad", AerialRobot, AerialRobotCfg())
task_registry.register("quad_with_obstacles", AerialRobotWithObstacles, AerialRobotWithObstaclesCfg())

task_registry.register( "X152b", X152bPx4, X152bPx4Cfg())