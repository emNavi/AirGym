import os
from airgym.utils.task_registry import task_registry

try:
    from .base.X152bPx4_config import X152bPx4Cfg
    from .base.X152bPx4 import X152bPx4
    task_registry.register("X152b", X152bPx4, X152bPx4Cfg())
except ImportError:
    print("Warning! X152bPx4_config or X152bPx4 cannot be imported. Ignore if using on real robot inference.")

try:
    from .base.X152bPx4_with_cam_config import X152bPx4WithCamCfg
    from .base.X152bPx4_with_cam import X152bPx4WithCam
    task_registry.register("X152b_with_cam", X152bPx4WithCam, X152bPx4WithCamCfg())
except ImportError:
    print("Warning! X152bPx4WithCamCfg or X152bPx4WithCam cannot be imported. Ignore if using on real robot inference.")

try:
    from .acrobatics.X152b_target_config import X152bTargetConfig
    from .acrobatics.X152b_target import X152bTarget
    task_registry.register("X152b_target", X152bTarget, X152bTargetConfig())
except ImportError:
    print("Warning! X152bTarget or X152bTargetConfig cannot be imported. Ignore if using on real robot inference.")

try:
    from .acrobatics.X152b_sigmoid_config import X152bSigmoidConfig
    from .acrobatics.X152b_sigmoid import X152bSigmoid
    task_registry.register("X152b_sigmoid", X152bSigmoid, X152bSigmoidConfig())
except ImportError:
    print("Warning! X152bSigmoid or X152bSigmoidConfig cannot be imported. Ignore if using on real robot inference.")