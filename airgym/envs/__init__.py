from airgym.envs.base.aerial_robot_config import AerialRobotCfg
from airgym.envs.base.aerial_robot_with_obstacles_config import AerialRobotWithObstaclesCfg
from .base.aerial_robot  import AerialRobot
from .base.aerial_robot_with_obstacles import AerialRobotWithObstacles

from .base.X152bPx4_config import X152bPx4Cfg
from .base.X152bPx4 import X152bPx4
from .base.X152bPx4_with_cam_config import X152bPx4WithCamCfg
from .base.X152bPx4_with_cam import X152bPx4WithCam

import os

from airgym.utils.task_registry import task_registry

task_registry.register( "quad", AerialRobot, AerialRobotCfg())
task_registry.register("quad_with_obstacles", AerialRobotWithObstacles, AerialRobotWithObstaclesCfg())

task_registry.register( "X152b", X152bPx4, X152bPx4Cfg())
task_registry.register( "X152b_with_cam", X152bPx4WithCam, X152bPx4WithCamCfg())