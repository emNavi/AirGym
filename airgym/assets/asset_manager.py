import os
import sys
import random
from airgym.assets import *

try:
    from isaacgym import gymapi, gymtorch
    print("isaacgym imported successful.")
except ImportError:
    print("isaacgym cannot be imported. Trying to import from gym_utils.")
    from airgym.utils.gym_utils import gymapi, gymtorch
    print("gymutil imported successful from gym_utils.")

class AssetManager:
    def __init__(self, config, device):
        self.cfg = config
        self.device = device
        self.asset_config = self.cfg.asset_config
        self.num_envs = self.cfg.env.num_envs

        self.include_robot = self.asset_config.include_robot
        self.include_single_asset = self.asset_config.include_single_asset
        self.include_group_asset = self.asset_config.include_group_asset
        self.include_boundary = self.asset_config.include_boundary
    
    def load_asset(self, gym, sim, env_handle, start_pose, env_id):
        self.gym = gym
        self.sim = sim
        
        robot_handles = []
        camera_handles = []
        camera_tensors = []
        env_asset_handles = [] # env_assets are assets except robots

        self.robot_counter = 0 # number of robots
        self.robot_num_bodies = 0 # number of robot bodies
        self.env_asset_count = 0 # number of env assets
        self.env_actor_count = 0 # all actors in the env including robots and assets
        self.env_asset_link_count = 0 # number of links of env assets
        self.env_boundary_count = 0 # number of boundaries like wall and ground
        self.segmentation_count = 0

        for k, v in self.include_robot.items():
            asset_name = k
            variant = None
            override_params = v
            asset_path, asset_info = registry.get_asset_info(asset_name, variant, override_params)
            
            asset_folder = os.path.dirname(asset_path)
            filename = os.path.basename(asset_path)
            asset_options = self._asset_options_gen(asset_info)
            if asset_info["enable_onboard_cameras"]:
                camera_props, local_transform = self._camera_props_gen(asset_info)

            robot = self.gym.load_asset(self.sim, asset_folder, filename, asset_options)

            self.robot_counter += asset_info["num_assets"]
            for idx in range(asset_info["num_assets"]):
                # insert robot asset
                robot_handle = self.gym.create_actor(env_handle, robot, start_pose, asset_name+'_'+str(idx), idx, asset_info["collision_mask"], 0)
                # append to lists
                robot_handles.append(robot_handle)
                self.env_actor_count += 1
                self.robot_num_bodies += self.gym.get_asset_rigid_body_count(robot)

                if asset_info["enable_onboard_cameras"]:
                    cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                    self.gym.attach_camera_to_body(cam_handle, env_handle, robot_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
                    camera_handles.append(cam_handle)
                    camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH)
                    torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor) # (height, width)
                    camera_tensors.append(torch_cam_tensor)

        env_asset_list = []

        for k, v in self.include_single_asset.items():
            asset_name = k.split('/')[0] if '/' in k else k
            variant = k.split('/')[-1] if '/' in k else None
            override_params = v
            asset_path, asset_info = registry.get_asset_info(asset_name, variant, override_params)
            
            asset_folder = os.path.dirname(asset_path)
            filename = os.path.basename(asset_path)
            asset_options = self._asset_options_gen(asset_info)
            
            asset_dict = {
                "asset_name": variant,
                "asset_folder": asset_folder,
                "asset_file_name": filename,
                "asset_options": asset_options,
                "body_semantic_label": asset_info['set_whole_body_semantic_mask'],
                "link_semantic_label": asset_info['set_semantic_mask_per_link'],
                "semantic_masked_links": asset_info['semantic_mask_link_list'],
                "semantic_id": asset_info['semantic_id'],
                "collision_mask": asset_info['collision_mask'],
                "color": asset_info['color'],
                "links_per_asset": asset_info['links_per_asset']
            }
            for i in range(asset_info["num_assets"]):
                env_asset_list.append(asset_dict)
                    
        for k, v in self.include_group_asset.items():
            asset_name = k.split('/')[0] if '/' in k else k
            variant = k.split('/')[-1] if '/' in k else None
            override_params = v
            assets, asset_info = registry.get_asset_info(asset_name, variant, override_params)
            asset_folder = os.path.dirname(assets) if variant else assets

            asset_options = self._asset_options_gen(asset_info)
            file_list = self._randomly_select_asset_files(asset_folder, asset_info["num_assets"])
            for i in range(len(file_list)):
                asset_dict = {
                    "asset_name": asset_name,
                    "asset_folder": asset_folder,
                    "asset_file_name": file_list[i],
                    "asset_options": asset_options,
                    "body_semantic_label": asset_info['set_whole_body_semantic_mask'],
                    "link_semantic_label": asset_info['set_semantic_mask_per_link'],
                    "semantic_masked_links": asset_info['semantic_mask_link_list'],
                    "semantic_id": asset_info['semantic_id'],
                    "collision_mask": asset_info['collision_mask'],
                    "color": asset_info['color'],
                    "links_per_asset": asset_info['links_per_asset']
                }
                env_asset_list.append(asset_dict)
        
        for k, v in self.include_boundary.items():
            asset_name = k
            variant = None
            override_params = v
            asset_path, asset_info = registry.get_asset_info(asset_name, variant, override_params)
            
            asset_folder = os.path.dirname(asset_path)
            filename = os.path.basename(asset_path)
            asset_options = self._asset_options_gen(asset_info)
            self.env_boundary_count += 1

            asset_dict = {
                "asset_name": asset_name,
                "asset_folder": asset_folder,
                "asset_file_name": filename,
                "asset_options": asset_options,
                "body_semantic_label": asset_info['set_whole_body_semantic_mask'],
                "link_semantic_label": asset_info['set_semantic_mask_per_link'],
                "semantic_masked_links": asset_info['semantic_mask_link_list'],
                "semantic_id": asset_info['semantic_id'],
                "collision_mask": asset_info['collision_mask'],
                "color": asset_info['color'],
                "links_per_asset": asset_info['links_per_asset']
            }
            for i in range(asset_info["num_assets"]):
                env_asset_list.append(asset_dict)
        
        for dict_item in env_asset_list:
            self.segmentation_count = max(self.segmentation_count, int(dict_item["semantic_id"])+1)

        self.asset_name_idx = {}
        for i in range(len(env_asset_list)):
            env_asset_handles.append(self._create_asset_from_file(env_asset_list[i], env_handle, start_pose, env_id))

        return robot_handles, camera_handles, camera_tensors, env_asset_handles
    

    def _create_asset_from_file(self, dict_item, env_handle, start_pose, env_id):
        """
        Inputs:
            dict_item: {
                "asset_name": asset_name,
                "asset_folder": asset_folder,
                "asset_file_name": filename,
                "asset_options": asset_options,
                "body_semantic_label": asset_info['body_semantic_label'],
                "link_semantic_label": asset_info['link_semantic_label'],
                "semantic_masked_links": asset_info['semantic_masked_links'],
                "semantic_id": asset_info['semantic_id'],
                "collision_mask": asset_info['collision_mask'],
                "color": asset_info['color']
                "links_per_asset": asset_info['links_per_asset']
            }
            env_handle: 
            start_pose:
            env_id: 

        """
        # load asset to isaacgym
        asset_name = dict_item["asset_name"]
        folder_path = dict_item["asset_folder"]
        filename = dict_item["asset_file_name"]
        asset_options = dict_item["asset_options"]
        whole_body_semantic = dict_item["body_semantic_label"]
        per_link_semantic = dict_item["link_semantic_label"]
        semantic_masked_links = dict_item["semantic_masked_links"]
        semantic_id = dict_item["semantic_id"]
        color = dict_item["color"]
        collision_mask = dict_item["collision_mask"]
        links_per_asset = dict_item["links_per_asset"]

        self.env_actor_count += 1
        self.env_asset_link_count += links_per_asset

        loaded_asset = self.gym.load_asset(self.sim, folder_path, filename, asset_options)

        assert not (whole_body_semantic and per_link_semantic)
        if semantic_id < 0:
            object_segmentation_id = self.segmentation_count
            self.segmentation_count += 1
        else:
            object_segmentation_id = semantic_id

        self.env_asset_count += 1
        if asset_name in self.asset_name_idx:
            self.asset_name_idx[asset_name] += 1
        else:
            self.asset_name_idx[asset_name] = 0

        # set asset handle
        env_asset_handle = self.gym.create_actor(env_handle, loaded_asset, start_pose, f"{asset_name}_"+str(self.asset_name_idx[asset_name]), env_id, collision_mask, object_segmentation_id)
        if len(self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)) > 1:
            print("Env asset has rigid body with more than 1 link: ", len(self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)))
            sys.exit(0)
        
        if per_link_semantic:
            rigid_body_names = None
            if len(semantic_masked_links) == 0:
                rigid_body_names = self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)
            else:
                rigid_body_names = semantic_masked_links
            for rb_index in range(len(rigid_body_names)):
                segmentation_count += 1
                self.gym.set_rigid_body_segmentation_id(env_handle, env_asset_handle, rb_index, segmentation_count)
        
        # set color according to semantic
        color_settig_list = [1, 2, 3, 7]
        if semantic_id in color_settig_list:
            if color is None:
                color = np.random.randint(low=50,high=200,size=3)

            self.gym.set_rigid_body_color(env_handle, env_asset_handle, 0, gymapi.MESH_VISUAL,
                    gymapi.Vec3(color[0]/255,color[1]/255,color[2]/255))
        
        return env_asset_handle
    

    def _asset_options_gen(self, asset_info):
        asset_options = gymapi.AssetOptions()
        items = [
            'collapse_fixed_joints',
            'replace_cylinder_with_capsule',
            'flip_visual_attachments',
            'fix_base_link',
            'density',
            'angular_damping',
            'linear_damping',
            'max_angular_velocity',
            'max_linear_velocity',
            'disable_gravity',
            'replace_cylinder_with_capsule',
            'vhacd_enabled',
        ]
        for item in items:
            setattr(asset_options, item, asset_info[item])
            if item == 'vhacd_enabled' and asset_info[item]:
                if 'vhacd_params.resolution' in asset_info:
                    asset_options.vhacd_params.resolution = asset_info['vhacd_params.resolution']
                if 'vhacd_params.max_convex_hulls' in asset_info:
                    asset_options.vhacd_params.max_convex_hulls = asset_info['vhacd_params.max_convex_hulls']
                if 'vhacd_params.max_num_vertices_per_ch' in asset_info:
                    asset_options.vhacd_params.max_num_vertices_per_ch = asset_info['vhacd_params.max_num_vertices_per_ch']

        return asset_options
    
    def _camera_props_gen(self, asset_info):
        camera_props = gymapi.CameraProperties()
        items = [
            'enable_tensors',
            'width',
            'height',
            'far_plane',
            'horizontal_fov',
            'use_collision_geometry',
        ]
        for item in items:
            setattr(camera_props, item, asset_info[item])

        local_transform = gymapi.Transform()
        # position of the camera relative to the body
        local_transform.p = gymapi.Vec3(*asset_info["local_transform.p"])
        # orientation of the camera relative to the body
        local_transform.r = gymapi.Quat(*asset_info["local_transform.r"])
        return camera_props, local_transform

    def _randomly_select_asset_files(self, asset_folder, num_files):
        file_name_list = [f for f in os.listdir(asset_folder)]
        urdf_files = []
        for f in file_name_list:
            if f.endswith('.urdf'):
                urdf_files.append(f)
            else:
                fm = os.path.join(f, "model.urdf")
                urdf_files.append(fm)
        selected_files = random.choices(urdf_files, k=num_files)
        return selected_files
    
    def get_env_asset_link_count(self):
        return self.env_asset_link_count

    def get_env_actor_count(self):
        return self.env_actor_count
    
    def get_env_asset_count(self):
        return self.env_asset_count
    
    def get_env_robot_count(self):
        return self.robot_counter

    def get_robot_num_bodies(self):
        return self.robot_num_bodies
    
    def get_env_boundary_count(self):
        return self.env_boundary_count
            