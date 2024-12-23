import os
from airgym.utils.task_registry import task_registry
import traceback

try:
    from .base.X152bPx4_config import X152bPx4Cfg
    from .base.X152bPx4 import X152bPx4
    task_registry.register("X152b", X152bPx4, X152bPx4Cfg())
except ImportError:
    print("WARNING! X152bPx4_config or X152bPx4 cannot be imported. Ignore if using on real robot inference.")
    traceback.print_exc()

try:
    from .base.X152bPx4_with_cam_config import X152bPx4WithCamCfg
    from .base.X152bPx4_with_cam import X152bPx4WithCam
    task_registry.register("X152b_with_cam", X152bPx4WithCam, X152bPx4WithCamCfg())
except ImportError:
    print("WARNING! X152bPx4WithCamCfg or X152bPx4WithCam cannot be imported. Ignore if using on real robot inference.")
    traceback.print_exc()
try:
    from .task.X152b_target_config import X152bTargetConfig
    from .task.X152b_target import X152bTarget
    task_registry.register("X152b_target", X152bTarget, X152bTargetConfig())
except ImportError:
    print("WARNING! X152bTarget or X152bTargetConfig cannot be imported. Ignore if using on real robot inference.")
    traceback.print_exc()
try:
    from .task.X152b_sigmoid_config import X152bSigmoidConfig
    from .task.X152b_sigmoid import X152bSigmoid
    task_registry.register("X152b_sigmoid", X152bSigmoid, X152bSigmoidConfig())
except ImportError:
    print("WARNING! X152bSigmoid or X152bSigmoidConfig cannot be imported. Ignore if using on real robot inference.")
    traceback.print_exc()