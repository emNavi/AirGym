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
    from airgym.envs.task.X152bPx4_planning_config import X152bPx4PlanningCfg
    from airgym.envs.task.X152bPx4_planning import X152bPx4Planning
    task_registry.register("X152b_planning", X152bPx4Planning, X152bPx4PlanningCfg())
except ImportError:
    print("WARNING! X152bPx4PlanningCfg or X152bPx4Planning cannot be imported. Ignore if using on real robot inference.")
    traceback.print_exc()

try:
    from .task.X152b_target_config import X152bTargetConfig
    from .task.X152b_target import X152bTarget
    task_registry.register("X152b_target", X152bTarget, X152bTargetConfig())
except ImportError:
    print("WARNING! X152bTarget or X152bTargetConfig cannot be imported. Ignore if using on real robot inference.")
    traceback.print_exc()

try:
    from .task.X152b_target_visual_config import X152bTargetVisualConfig
    from .task.X152b_target_visual import X152bTargetVisual
    task_registry.register("X152b_target_visual", X152bTargetVisual, X152bTargetVisualConfig())
except ImportError:
    print("WARNING! X152bTargetVisual or X152bTargetVisualConfig cannot be imported. Ignore if using on real robot inference.")
    traceback.print_exc()

try:
    from .task.X152b_slit_config import X152bSlitConfig
    from .task.X152b_slit import X152bSlit
    task_registry.register("X152b_slit", X152bSlit, X152bSlitConfig())
except ImportError:
    print("WARNING! X152bSlit or X152bSlitConfig cannot be imported. Ignore if using on real robot inference.")
    traceback.print_exc()

try:
    from .task.X152b_avoid_config import X152bAvoidConfig
    from .task.X152b_avoid import X152bAvoid
    task_registry.register("X152b_avoid", X152bAvoid, X152bAvoidConfig())
except ImportError:
    print("WARNING! X152bAvoid or X152bAvoidConfig cannot be imported. Ignore if using on real robot inference.")
    traceback.print_exc()
try:
    from .task.X152b_sin_config import X152bSinConfig
    from .task.X152b_sin import X152bSin
    task_registry.register("X152b_sin", X152bSin, X152bSinConfig())
except ImportError:
    print("WARNING! X152bSin or X152bSinConfig cannot be imported. Ignore if using on real robot inference.")
    traceback.print_exc()