import os
import sys
import random
import time
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
    
    def load_asset(self, gym, sim):
        self.gym = gym
        self.sim = sim
        self._update_asset_param()

        self.robot_list = []
        self.asset_list = []

        self.robot_counter = 0 # number of robots
        self.robot_num_bodies = 0 # number of robot bodies
        self.env_asset_count = 0 # number of env assets
        self.env_actor_count = 0 # all actors in the env including robots and assets
        self.env_asset_link_count = 0 # number of links of env assets
        self.env_boundary_count = 0 # number of boundaries like wall and ground

        for k in self.include_robot.keys():
            robot_name = k
            variant = None
            asset_params = self.param_record[k]
            asset_path = asset_params["path"]
            
            asset_folder = os.path.dirname(asset_path)
            filename = os.path.basename(asset_path)
            asset_options = self._asset_options_gen(asset_params)
            gym_robot = self.gym.load_asset(self.sim, asset_folder, filename, asset_options)
            robot_idx = 0

            if asset_params["enable_onboard_cameras"]:
                camera_props, local_transform = self._camera_props_gen(asset_params)
            else:
                camera_props = local_transform = None

            for i in range(asset_params["num_assets"]):
                robot_dict = {
                    "robot_name": robot_name,
                    "robot_idx": robot_idx,
                    "asset_options": asset_options,
                    "semantic_id": asset_params['semantic_id'],
                    "collision_mask": asset_params['collision_mask'],
                    "color": asset_params['color'],
                    "enable_onboard_cameras": asset_params['enable_onboard_cameras'],
                    "camera_props": camera_props,
                    "local_transform": local_transform,
                    "gym_robot": gym_robot,
                }
                self.robot_list.append(robot_dict)

                robot_idx += 1
                self.robot_counter += 1
                self.env_actor_count += 1
                self.robot_num_bodies += self.gym.get_asset_rigid_body_count(gym_robot)

        for k in self.include_single_asset.keys():
            asset_name = k.split('/')[0] if '/' in k else k
            variant = k.split('/')[-1] if '/' in k else None
            asset_params = self.param_record[k]
            asset_path = asset_params["path"]

            asset_folder = os.path.dirname(asset_path)
            filename = os.path.basename(asset_path)
            asset_options = self._asset_options_gen(asset_params)
            gym_env_asset = self.gym.load_asset(self.sim, asset_folder, filename, asset_options)

            asset_idx = 0
            for i in range(asset_params["num_assets"]):
                asset_dict = {
                    "asset_name": asset_name,
                    "asset_idx": asset_idx,
                    "variant": variant,
                    "asset_options": asset_options,
                    "body_semantic_label": asset_params['set_whole_body_semantic_mask'],
                    "semantic_id": asset_params['semantic_id'],
                    "collision_mask": asset_params['collision_mask'],
                    "color": asset_params['color'],
                    "links_per_asset": asset_params['links_per_asset'],
                    "gym_env_asset": gym_env_asset,
                }
                self.asset_list.append(asset_dict)

                asset_idx += 1
                self.env_asset_count += 1
                self.env_actor_count += 1
                self.env_asset_link_count += asset_params['links_per_asset']
        
        for k in self.include_group_asset.keys():
            assert '/' not in k, "Group asset cannot be specified, please use include_single_asset!"
            asset_name = k
            variant = None
            asset_params = self.param_record[k]
            assert os.path.isdir(asset_params["path"])
            
            file_name_list = [f for f in os.listdir(asset_params["path"])]
            gym_env_asset_list = []
            for f in file_name_list:
                if f.lower().endswith('.urdf'):
                    asset_folder = asset_params["path"]
                    filename = f
                else:
                    asset_folder = os.path.join(asset_params["path"], f)
                    filename = "model.urdf"
                asset_options = self._asset_options_gen(asset_params)
                if asset_name == "trees":
                    print(asset_folder, filename)
                gym_env_asset = self.gym.load_asset(self.sim, asset_folder, filename, asset_options)
                gym_env_asset_list.append(gym_env_asset)
            
            asset_idx = 0
            for i in range(asset_params["num_assets"]):
                asset_dict = {
                    "asset_name": asset_name,
                    "asset_idx": asset_idx,
                    "variant": variant,
                    "asset_options": asset_options,
                    "body_semantic_label": asset_params['set_whole_body_semantic_mask'],
                    "semantic_id": asset_params['semantic_id'],
                    "collision_mask": asset_params['collision_mask'],
                    "color": asset_params['color'],
                    "links_per_asset": asset_params['links_per_asset'],
                    "gym_env_asset": random.choice(gym_env_asset_list),
                }
                self.asset_list.append(asset_dict)

                asset_idx += 1
                self.env_asset_count += 1
                self.env_actor_count += 1
                self.env_asset_link_count += asset_params['links_per_asset']

        for k in self.include_boundary.keys():
            asset_name = k
            variant = None
            asset_params = self.param_record[k]
            asset_path = asset_params["path"]
            
            asset_folder = os.path.dirname(asset_path)
            filename = os.path.basename(asset_path)
            asset_options = self._asset_options_gen(asset_params)
            boundary = self.gym.load_asset(self.sim, asset_folder, filename, asset_options)

            asset_idx = 0
            for i in range(asset_params["num_assets"]):
                boundary_dict = {
                    "asset_name": asset_name,
                    "asset_idx": asset_idx,
                    "variant": variant,
                    "asset_options": asset_options,
                    "body_semantic_label": asset_params['set_whole_body_semantic_mask'],
                    "semantic_id": asset_params['semantic_id'],
                    "collision_mask": asset_params['collision_mask'],
                    "color": asset_params['color'],
                    "links_per_asset": asset_params['links_per_asset'],
                    "gym_env_asset": boundary,
                }
                self.asset_list.append(boundary_dict)

                asset_idx += 1
                self.env_asset_count += 1
                self.env_actor_count += 1
                self.env_boundary_count += 1
                self.env_asset_link_count += asset_params['links_per_asset']

    def create_asset(self, env_handle, start_pose, env_id):        
        robot_handles = []
        camera_handles = []
        camera_tensors = []
        env_asset_handles = [] # env_assets are assets except robots
        body_handle = 0
        rigid_body_count = 0

        for robot_dict in self.robot_list:
            gym_robot = robot_dict["gym_robot"]
            robot_name = robot_dict["robot_name"]
            robot_idx = robot_dict["robot_idx"]
            # insert robot asset
            robot_handle = self.gym.create_actor(env_handle, gym_robot, start_pose, robot_name+'_'+str(robot_idx), env_id, robot_dict["collision_mask"], 0)
            # append to lists
            robot_handles.append(robot_handle)

            if robot_dict["enable_onboard_cameras"]:
                cam_handle = self.gym.create_camera_sensor(env_handle, robot_dict["camera_props"])
                # 这里需要将camera attach到对应的刚体上,而不是robot_handle上, X152b有5个刚体, 0号刚体是base_link
                # self.gym.attach_camera_to_body(cam_handle, env_handle, robot_handle, robot_dict["local_transform"], gymapi.FOLLOW_TRANSFORM)
                self.gym.attach_camera_to_body(cam_handle, env_handle, body_handle+rigid_body_count, robot_dict["local_transform"], gymapi.FOLLOW_TRANSFORM)
                camera_handles.append(cam_handle)
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor) # (height, width)
                camera_tensors.append(torch_cam_tensor)
                rigid_body_count += self.gym.get_actor_rigid_body_count(env_handle, robot_handle)

        for env_asset_dict in self.asset_list:
            env_asset_handles.append(self._create_asset_from_file(env_asset_dict, env_handle, start_pose, env_id))
        
        return robot_handles, camera_handles, camera_tensors, env_asset_handles
    
    def _update_asset_param(self):
        self.param_record = {}

        def _override_params(params, asset_type, variant, override_params):
            params_cp = params.copy()

            if variant is not None and asset_type == 'group':
                params_cp["path"] = os.path.join(params["path"], variant, "model.urdf")
                assert os.path.exists(params_cp["path"]), f"Variant named {asset_name}/{variant} does not exist."
            if variant is not None and asset_type == 'single':
                raise ValueError(f"Single asset {asset_name} has no variant.")

            if override_params:
                params_cp.update(override_params)

            return params_cp

        for k, v, in self.include_robot.items():
            asset_name = k
            variant = None
            override_params = v
            self.param_record[k] = _override_params(*registry.get_asset_info(asset_name), variant, override_params)

        for k, v in self.include_single_asset.items():
            asset_name = k.split('/')[0] if '/' in k else k
            variant = k.split('/')[-1] if '/' in k else None
            override_params = v
            self.param_record[k] = _override_params(*registry.get_asset_info(asset_name), variant, override_params)

        for k, v in self.include_group_asset.items():
            asset_name = k.split('/')[0] if '/' in k else k
            variant = k.split('/')[-1] if '/' in k else None
            override_params = v
            self.param_record[k] = _override_params(*registry.get_asset_info(asset_name), variant, override_params)

        for k, v in self.include_boundary.items():
            asset_name = k
            variant = None
            override_params = v
            self.param_record[k] = _override_params(*registry.get_asset_info(asset_name), variant, override_params)

    def _create_asset_from_file(self, dict_item, env_handle, start_pose, env_id):
        """
        Inputs:
            dict_item: {
                "asset_name": asset_name,
                "asset_idx": asset_idx,
                "variant": variant,
                "asset_options": asset_options,
                "body_semantic_label": asset_params['set_whole_body_semantic_mask'],
                "semantic_id": asset_params['semantic_id'],
                "collision_mask": asset_params['collision_mask'],
                "color": asset_params['color'],
                "links_per_asset": asset_params['links_per_asset'],
                "gym_env_asset": gym_env_asset,
            }
            env_handle: 
            start_pose:
            env_id: 

        """
        # load asset to isaacgym
        asset_name = dict_item["asset_name"]
        asset_idx = dict_item["asset_idx"]
        variant = dict_item['variant']
        semantic_id = dict_item["semantic_id"]
        color = dict_item["color"]
        collision_mask = dict_item["collision_mask"]
        gym_env_asset = dict_item['gym_env_asset']
        assert semantic_id != 0

        # set asset handle
        env_asset_handle = self.gym.create_actor(env_handle, gym_env_asset, start_pose, f"{asset_name}_{asset_idx}", env_id, collision_mask, semantic_id)
        if len(self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)) > 1:
            print("Env asset has rigid body with more than 1 link: ", len(self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)))
            sys.exit(0)
        
        # set color according to semantic
        color_settig_list = [1, 2, 3, 7]
        if semantic_id in color_settig_list:
            if color is None:
                color = np.random.randint(low=50,high=200,size=3)

            self.gym.set_rigid_body_color(env_handle, env_asset_handle, 0, gymapi.MESH_VISUAL,
                    gymapi.Vec3(color[0]/255,color[1]/255,color[2]/255))
        
        return env_asset_handle
    

    def _asset_options_gen(self, asset_param):
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
            setattr(asset_options, item, asset_param[item])
            if item == 'vhacd_enabled' and asset_param[item]:
                if 'vhacd_params.resolution' in asset_param:
                    asset_options.vhacd_params.resolution = asset_param['vhacd_params.resolution']
                if 'vhacd_params.max_convex_hulls' in asset_param:
                    asset_options.vhacd_params.max_convex_hulls = asset_param['vhacd_params.max_convex_hulls']
                if 'vhacd_params.max_num_vertices_per_ch' in asset_param:
                    asset_options.vhacd_params.max_num_vertices_per_ch = asset_param['vhacd_params.max_num_vertices_per_ch']

        return asset_options
    
    def _camera_props_gen(self, asset_param):
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
            setattr(camera_props, item, asset_param[item])

        local_transform = gymapi.Transform()
        # position of the camera relative to the body
        local_transform.p = gymapi.Vec3(*asset_param["local_transform.p"])
        # orientation of the camera relative to the body
        local_transform.r = gymapi.Quat(*asset_param["local_transform.r"])
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
            