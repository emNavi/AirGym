
from .base.X152bPx4_config import X152bPx4Cfg
from .base.X152bPx4 import X152bPx4
from .base.X152bPx4_with_cam_config import X152bPx4WithCamCfg
from .base.X152bPx4_with_cam import X152bPx4WithCam

import os

from airgym.utils.task_registry import task_registry

task_registry.register("X152b", X152bPx4, X152bPx4Cfg())
task_registry.register("X152b_with_cam", X152bPx4WithCam, X152bPx4WithCamCfg())

from .base.inference import inference
task_registry.register("inference", inference, X152bPx4Cfg())