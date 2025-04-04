import os
from typing import Dict, Any, Optional, Union, List
import json

try:
    from isaacgym import gymapi
    print("isaacgym imported successful.")
except ImportError:
    print("isaacgym cannot be imported. Trying to import from gym_utils.")
    from airgym.utils.gym_utils import gymapi
    print("gymutil imported successful from gym_utils.")

DEFAULT_PARAMS = {
    'base_link_name': "base_link",
    'foot_name': None,
    'penalize_contacts_on': [],
    'terminate_after_contacts_on': [],
    'armature': 0.001,
    'collapse_fixed_joints': True,
    'replace_cylinder_with_capsule': False,
    'flip_visual_attachments': False,
    'fix_base_link': True,
    'density': -1,
    'collision_mask': 1, # objects with the same collision mask will not collide
    'angular_damping': 0.,
    'linear_damping': 0.,
    'max_angular_velocity': 100.,
    'max_linear_velocity': 100.,
    'disable_gravity': False,
    'vhacd_enabled': False,
    'color': None,
    'num_assets': 1,
}

class AssetRegistry:
    def __init__(self):
        self._assets = {}
    
    def register_asset(self,
                      asset_name: str,
                      override_params: Dict[str, Any],
                      asset_type: str = "single"):

        if asset_name in self._assets:
            raise ValueError(f"Asset {asset_name} already registered")
        
        if asset_type not in ["single", "group"]:
            raise ValueError(f"Invalid asset_type: {asset_type}. Must be 'single' or 'group'")
        
        params = DEFAULT_PARAMS.copy()
        params.update(override_params)

        self._assets[asset_name] = {
            "params": params,
            "type": asset_type
        }
    
    def get_asset_info(self,
                 asset_name: str,
                 variant: Optional[str] = None,
                 override_params: Optional[Dict[str, Any]] = None) -> Any:

        if asset_name not in self._assets:
            raise KeyError(f"Asset {asset_name} not registered")
        
        asset_info = self._assets[asset_name]
        asset_type = asset_info["type"]
        params = asset_info["params"].copy()
        
        if variant is not None and asset_type == 'group':
            asset_path = os.path.join(params["path"], f"{variant}.urdf")
            asset_path = asset_path if os.path.exists(asset_path) else os.path.join(params['path'], variant, 'model.urdf')
            assert os.path.exists(asset_path), f"Variant named {asset_name}/{variant} does not exist."
        elif variant is not None and asset_type == 'single':
            raise ValueError(f"Single asset {asset_name} has no variant.")
        elif not variant:
            asset_path = params["path"]
        
        if override_params:
            params.update(override_params)

        return asset_path, params
    
    def get_variants(self, asset_name: str) -> List[str]:
        if asset_name not in self._assets:
            raise KeyError(f"Asset {asset_name} not registered.")
        assert self._assets[asset_name]['type'] == 'group', f"Asset {asset_name} has no variants."
        
        asset_info = self._assets[asset_name]
        
        path = self._resolve_env_vars(asset_info["params"]["path"])
        if not os.path.exists(path):
            return []
        
        variants = []
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                variants.append(entry)
            elif entry.endswith('.urdf'):
                variants.append(os.path.splitext(entry)[0])
        
        return variants
    
    def _resolve_env_vars(self, path: str) -> str:
        if "{AIRGYM_ROOT_DIR}" in path:
            airgym_root = os.getenv("AIRGYM_ROOT_DIR")
            if not airgym_root:
                raise ValueError("AIRGYM_ROOT_DIR environment variable not set")
            path = path.replace("{AIRGYM_ROOT_DIR}", airgym_root)
        
        return path