import os
import random

try:
    from isaacgym import gymapi
    print("isaacgym imported successful.")
except ImportError:
    print("isaacgym cannot be imported. Trying to import from gym_utils.")
    from airgym.utils.gym_utils import gymapi
    print("gymutil imported successful from gym_utils.")


# from isaacgym.torch_utils import quat_from_euler_xyz

import torch
# import pytorch3d.transforms as p3d_transforms

from .asset_register import *

def asset_class_to_AssetOptions(asset_class):
    asset_options = gymapi.AssetOptions()
    asset_options.collapse_fixed_joints = asset_class.collapse_fixed_joints
    asset_options.replace_cylinder_with_capsule = asset_class.replace_cylinder_with_capsule
    asset_options.flip_visual_attachments = asset_class.flip_visual_attachments
    asset_options.fix_base_link = asset_class.fix_base_link
    asset_options.density = asset_class.density
    asset_options.angular_damping = asset_class.angular_damping
    asset_options.linear_damping = asset_class.linear_damping
    asset_options.max_angular_velocity = asset_class.max_angular_velocity
    asset_options.max_linear_velocity = asset_class.max_linear_velocity
    asset_options.disable_gravity = asset_class.disable_gravity
    asset_options.replace_cylinder_with_capsule = True

    # for object convex decomposition
    asset_options.vhacd_enabled = asset_class.vhacd_enabled
    if asset_options.vhacd_enabled:
        asset_options.vhacd_params.resolution = asset_class.resolution
    return asset_options


class AssetManager:
    def __init__(self, config, device):
        self.cfg = config
        self.device = device
        self.asset_config = self.cfg.asset_config
        self.assets = []
        self.asset_pose_tensor = None
        self.asset_const_inv_mask_tensor = None
        self.asset_min_state_tensor = None
        self.asset_max_state_tensor = None
        self.num_envs = self.cfg.env.num_envs
        self.env_actor_count = 0
        self.env_link_count = 0
        
        self.env_lower_bound_min = torch.tensor(self.asset_config.env_lower_bound_min, device=self.device, requires_grad=False)
        self.env_lower_bound_max = torch.tensor(self.asset_config.env_lower_bound_max, device=self.device, requires_grad=False)
        self.env_upper_bound_min = torch.tensor(self.asset_config.env_upper_bound_min, device=self.device, requires_grad=False)
        self.env_upper_bound_max = torch.tensor(self.asset_config.env_upper_bound_max, device=self.device, requires_grad=False)

        self.env_lower_bound_diff = self.env_lower_bound_max - self.env_lower_bound_min
        self.env_upper_bound_diff = self.env_upper_bound_max - self.env_upper_bound_min

        # initialize the environmental bounds for randomization
        self.env_lower_bound = self.env_lower_bound_min.repeat(self.num_envs, 1)
        self.env_upper_bound = self.env_upper_bound_max.repeat(self.num_envs, 1)
        self.env_bound_diff = torch.zeros((self.num_envs,3), device=self.device, requires_grad=False)

        self.asset_type_to_dict_map = self.asset_config.asset_type_to_dict_map
        
        self.load_asset_tensors()
        self.randomize_pose()


    def _add_asset_2_tensor(self, asset_class):

        self.env_actor_count += asset_class.num_assets
        self.env_link_count += asset_class.num_assets * asset_class.links_per_asset
        
        # Define the asset tensors together for the number of assets of the same class being loaded
        asset_tensor = torch.zeros((1,6), dtype=torch.float, device=self.device).expand(1,-1)
        
        asset_tensor = asset_tensor.tile(asset_class.num_assets, 1)
        min_state_tensor = torch.tensor((asset_class.min_position_ratio + asset_class.min_euler_angles), dtype=torch.float, device=self.device).expand(asset_class.num_assets,-1)
        max_state_tensor = torch.tensor((asset_class.max_position_ratio + asset_class.max_euler_angles), dtype=torch.float, device=self.device).expand(asset_class.num_assets,-1)
        specified_state_tensor = torch.tensor((asset_class.specified_position + asset_class.specified_euler_angle), dtype=torch.float, device=self.device).expand(asset_class.num_assets,-1)
        
        # If the whole global asset pose tensor is not defined, define it and then append more copies to it
        if self.asset_pose_tensor is None:
            self.asset_pose_tensor = asset_tensor
            self.asset_min_state_tensor = min_state_tensor
            self.asset_max_state_tensor = max_state_tensor
            self.asset_specified_state_tensor = specified_state_tensor
        # if the tensor exists, append copies to it.
        else:
            self.asset_pose_tensor = torch.vstack(
                (self.asset_pose_tensor, asset_tensor))
            self.asset_min_state_tensor = torch.vstack(
                (self.asset_min_state_tensor, min_state_tensor))
            self.asset_max_state_tensor = torch.vstack(
                (self.asset_max_state_tensor, max_state_tensor))
            self.asset_specified_state_tensor = torch.vstack(
                (self.asset_specified_state_tensor, specified_state_tensor))


    def load_asset_tensors(self):
        # Pre-load the tensors before the assets are created
        for asset_key, include_asset in self.asset_config.include_asset_type.items():
            if not include_asset:
                continue
            print("Adding assets group: {}".format(asset_key))
            asset_class = self.asset_type_to_dict_map[asset_key]
            self._add_asset_2_tensor(asset_class)
                
        for asset_key, include_asset in self.asset_config.include_specific_asset.items():
            if not include_asset:
                continue
            print("Adding specific assets: {}".format(asset_key))
            asset_class = self.asset_type_to_dict_map[asset_key]
            self._add_asset_2_tensor(asset_class)
        
        if self.asset_pose_tensor is None:
            return

        self.asset_pose_tensor = torch.tile(
            self.asset_pose_tensor.unsqueeze(0), (self.cfg.env.num_envs, 1, 1))
        self.asset_min_state_tensor = self.asset_min_state_tensor.expand(self.cfg.env.num_envs, -1, -1)
        self.asset_max_state_tensor = self.asset_max_state_tensor.expand(self.cfg.env.num_envs, -1, -1)
        

    def prepare_assets_for_simulation(self, gym, sim):
        asset_list = []
        for asset_key, include_asset in self.asset_config.include_asset_type.items():
            if not include_asset:
                continue
            asset_class = self.asset_type_to_dict_map[asset_key]
            asset_options = asset_class_to_AssetOptions(asset_class)

            semantic_masked_links = asset_class.semantic_mask_link_list
            semantic_id = asset_class.semantic_id
            collision_mask = asset_class.collision_mask
            body_semantic_label = asset_class.set_whole_body_semantic_mask
            link_semantic_label = asset_class.set_semantic_mask_per_link
            if not (body_semantic_label or link_semantic_label):
                semantic_id = -1

            # "Only one of body_semantic_label and link_semantic_label can be True"
            assert not (body_semantic_label and link_semantic_label)

            color = asset_class.color

            folder_path = os.path.join(
                self.asset_config.folder_path, asset_key)
            file_list = self.randomly_select_asset_files(
                folder_path, asset_class.num_assets)

            for file_name in file_list:
                asset_dict = {
                    "asset_folder_path": folder_path,
                    "asset_file_name": file_name,
                    "asset_options": asset_options,
                    "body_semantic_label": body_semantic_label,
                    "link_semantic_label": link_semantic_label,
                    "semantic_masked_links": semantic_masked_links,
                    "semantic_id": semantic_id,
                    "collision_mask": collision_mask,
                    "color": color
                }
                asset_list.append(asset_dict)
        
        for asset_key, include_asset in self.asset_config.include_specific_asset.items():
            if not include_asset:
                continue
            assert "/" in asset_key, "The asset_key should be a specific asset!"
            asset_class = self.asset_type_to_dict_map[asset_key]
            asset_options = asset_class_to_AssetOptions(asset_class)

            semantic_masked_links = asset_class.semantic_mask_link_list
            semantic_id = asset_class.semantic_id
            collision_mask = asset_class.collision_mask
            body_semantic_label = asset_class.set_whole_body_semantic_mask
            link_semantic_label = asset_class.set_semantic_mask_per_link
            if not (body_semantic_label or link_semantic_label):
                semantic_id = -1

            # "Only one of body_semantic_label and link_semantic_label can be True"
            assert not (body_semantic_label and link_semantic_label)

            color = asset_class.color

            assert asset_class.num_assets == 1, "Only one asset can be loaded if the asset_key is a specific asset!"
            # assert hasattr(asset_class, "specified_position") and \
            #     all(v > -900 for v in asset_class.specified_position) and \
            #         all(v > -900 for v in asset_class.specified_euler_angle), \
            #             "The specified position and angular must be defined for the specific asset!"
            
            asset_type = asset_key.split("/")[0]
            asset_key = asset_key.split("/")[-1]
            folder_path = os.path.join(
                self.asset_config.folder_path, asset_type, asset_key)
            file_list = []

            file_name_list = [f for f in os.listdir(folder_path)]
            if os.path.exists(folder_path + ".urdf"):
                file_list.append(folder_path + ".urdf")
            else:
                for f in file_name_list:
                    if f.endswith('.urdf'):
                        file_list.append(f)
                assert len(file_list) == 1, f"{len(file_list)} urdf file is added! Only one urdf file should be present in the folder!"

            for file_name in file_list:
                asset_dict = {
                    "asset_folder_path": folder_path,
                    "asset_file_name": file_name,
                    "asset_options": asset_options,
                    "body_semantic_label": body_semantic_label,
                    "link_semantic_label": link_semantic_label,
                    "semantic_masked_links": semantic_masked_links,
                    "semantic_id": semantic_id,
                    "collision_mask": collision_mask,
                    "color": color
                }
                asset_list.append(asset_dict)
        
        return asset_list

    def randomly_select_asset_files(self, folder_path, num_files):
        file_name_list = [f for f in os.listdir(folder_path)]
        urdf_files = []
        for f in file_name_list:
            if f.endswith('.urdf'):
                urdf_files.append(f)
            else:
                fm = os.path.join(f, "model.urdf")
                urdf_files.append(fm)
        selected_files = random.choices(urdf_files, k=num_files)
        return selected_files
    
    def randomize_pose(self, num_obstacles = None, reset_envs = None):
        if self.asset_pose_tensor is None:
            return

        # Sampled environment bounds
        self.env_lower_bound = torch.rand((self.num_envs,3), device=self.device, requires_grad=False) * self.env_lower_bound_diff + self.env_lower_bound_min
        self.env_upper_bound = torch.rand((self.num_envs,3), device=self.device, requires_grad=False) * self.env_upper_bound_diff + self.env_upper_bound_min

        self.env_bound_diff = (self.env_upper_bound - self.env_lower_bound)

        pos_ratio_euler_asbolute = self.asset_min_state_tensor + torch.rand_like(self.asset_min_state_tensor)*(self.asset_max_state_tensor - self.asset_min_state_tensor)
        self.asset_pose_tensor[:, :, :3] = self.env_lower_bound.unsqueeze(1) + self.env_bound_diff.unsqueeze(1) * pos_ratio_euler_asbolute[:,:,:3]
        
        self.asset_pose_tensor[:, :, 3:6] = pos_ratio_euler_asbolute[:, :, 3:6]

        self.asset_pose_tensor = torch.where(self.asset_specified_state_tensor > -900, self.asset_specified_state_tensor, self.asset_pose_tensor)
        return
        
    def get_env_link_count(self):
        return self.env_link_count

    def get_env_actor_count(self):
        return self.env_actor_count