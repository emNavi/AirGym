import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET

from airgym import AIRGYM_ROOT_DIR, AIRGYM_ROOT_DIR

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from airgym.envs.base.X152bPx4 import X152bPx4
import airgym.utils.rotations as rot_utils
from airgym.envs.task.X152b_avoid_config import X152bAvoidConfig
from airgym.utils.asset_manager import AssetManager

from rlPx4Controller.pyParallelControl import ParallelRateControl,ParallelVelControl,ParallelAttiControl,ParallelPosControl

import matplotlib.pyplot as plt
from airgym.utils.helpers import asset_class_to_AssetOptions
from airgym.utils.rotations import quats_to_euler_angles
import time

import pytorch3d.transforms as T

def quaternion_conjugate(q: torch.Tensor):
    """Compute the conjugate of a quaternion."""
    q_conj = q.clone()
    q_conj[:, :3] = -q_conj[:, :3]
    return q_conj

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor):
    """Multiply two quaternions."""
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack((x, y, z, w), dim=-1)

class X152bAvoid(X152bPx4):

    def __init__(self, cfg: X152bAvoidConfig, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        print("ctl mode =========== ",cfg.env.ctl_mode)
        self.ctl_mode = cfg.env.ctl_mode
        self.cfg.env.num_actions = 5 if cfg.env.ctl_mode == "atti" else 4
        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.debug_viz = False
        num_actors = 1

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless

        self.env_asset_manager = AssetManager(self.cfg, sim_device)

        super(X152bPx4, self).__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        self.contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        num_actors = self.env_asset_manager.get_env_actor_count() + 1 # Number of obstacles in the environment + one robot
        bodies_per_env = self.env_asset_manager.get_env_link_count() + self.robot_num_bodies # Number of links in the environment + robot

        self.vec_root_tensor = gymtorch.wrap_tensor(
            self.root_tensor).view(self.num_envs, num_actors, 13)

        self.root_states = self.vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[..., 0:3]
        self.root_quats = self.root_states[..., 3:7] # x,y,z,w
        self.root_linvels = self.root_states[..., 7:10]
        self.root_angvels = self.root_states[..., 10:13]

        self.privileged_obs_buf = None
        if self.vec_root_tensor.shape[1] > 1:
            self.env_asset_root_states = self.vec_root_tensor[:, 1:, :]
            if self.get_privileged_obs:
                self.privileged_obs_buf = self.env_asset_root_states
                
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.counter = 0

        # get states of cube
        self.cube_states = self.env_asset_root_states[:, 1, :]
        self.cube_positions = self.cube_states[..., 0:3]
        self.cube_quats = self.cube_states[..., 3:7] # x,y,z,w
        self.cube_linvels = self.cube_states[..., 7:10]
        self.cube_angvels = self.cube_states[..., 10:13]

        # controller
        self.cmd_thrusts = torch.zeros((self.num_envs, 4))
        # choice 1 from rate ctrl and vel ctrl
        if(cfg.env.ctl_mode == "pos"):
            self.action_upper_limits = torch.tensor(
            [6, 6, 6, 6.0], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor(
            [-6, -6, -6, -6.0], device=self.device, dtype=torch.float32)
            self.parallel_pos_control = ParallelPosControl(self.num_envs)
        elif(cfg.env.ctl_mode == "vel"):
            self.action_upper_limits = torch.tensor(
                [6, 6, 6, 6], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor(
                [-6, -6, -6, -6], device=self.device, dtype=torch.float32)
            self.parallel_vel_control = ParallelVelControl(self.num_envs)
        elif(cfg.env.ctl_mode == "atti"):
            self.action_upper_limits = torch.tensor(
            [1, 1, 1, 1, 1], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor(
            [-1, -1, -1, -1, 0.], device=self.device, dtype=torch.float32)
            self.parallel_atti_control = ParallelAttiControl(self.num_envs)
        elif(cfg.env.ctl_mode == "rate"):
            self.action_upper_limits = torch.tensor(
                [6, 6, 6, 1], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor(
                [-6, -6, -6, 0], device=self.device, dtype=torch.float32)
            self.parallel_rate_control = ParallelRateControl(self.num_envs)
        elif(cfg.env.ctl_mode == "prop"):
            self.action_upper_limits = torch.tensor(
                [1, 1, 1, 1], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor(
                [0, 0, 0, 0], device=self.device, dtype=torch.float32)
        else:
            print("Mode Error!")

        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3),
                                  dtype=torch.float32, device=self.device, requires_grad=False)
        self.torques = torch.zeros((self.num_envs, bodies_per_env, 3),
                                   dtype=torch.float32, device=self.device, requires_grad=False)
        
        # control parameters
        self.thrusts = torch.zeros((self.num_envs, 4, 3), dtype=torch.float32, device=self.device)
        self.thrust_cmds_damp = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.thrust_rot_damp = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

        # actions
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.pre_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        # reward integration buffers
        self.int_pos_error = torch.zeros((self.num_envs, 10), device=self.device)
        self.int_yaw_error = torch.zeros((self.num_envs, 10), device=self.device)

        self.pre_root_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.pre_root_angvels = torch.zeros((self.num_envs, 3), device=self.device)

        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env
            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        if self.cfg.use_tcn:
            self.tcn_seqs_len = self.cfg.tcn_seqs_len
            self.obs_seqs_buf = torch.zeros(
                (self.num_envs, self.tcn_seqs_len, self.cfg.env.num_observations), device=self.device, dtype=torch.float32)
        
        self.flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    def _create_envs(self):
        print("\n\n\n\n\n CREATING ENVIRONMENT \n\n\n\n\n\n")
        asset_path = self.cfg.asset_config.X152b.file.format(
            AIRGYM_ROOT_DIR=AIRGYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = asset_class_to_AssetOptions(self.cfg.asset_config.X152b)

        X152b = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)

        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(X152b)

        start_pose = gymapi.Transform()
        # create env instance
        pos = torch.tensor([0, 0, 0], device=self.device)
        start_pose.p = gymapi.Vec3(*pos)
        self.env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-self.env_spacing, -
                                self.env_spacing, -self.env_spacing)
        env_upper = gymapi.Vec3(
            self.env_spacing, self.env_spacing, self.env_spacing)
        self.actor_handles = []
        self.env_asset_handles = []
        self.envs = []

        self.segmentation_counter = 0

        for i in range(self.num_envs):
            # create environment
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # insert robot asset
            actor_handle = self.gym.create_actor(env_handle, X152b, start_pose, "robot", i, self.cfg.asset_config.X152b.collision_mask, 0)
            # append to lists
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            env_asset_list = self.env_asset_manager.prepare_assets_for_simulation(self.gym, self.sim)
            asset_counter = 0

            # have the segmentation counter be the max defined semantic id + 1. Use this to set the semantic mask of objects that are
            # do not have a defined semantic id in the config file, but still requre one. Increment for every instance in the next snippet
            for dict_item in env_asset_list:
                self.segmentation_counter = max(self.segmentation_counter, int(dict_item["semantic_id"])+1)

            for dict_item in env_asset_list:
                folder_path = dict_item["asset_folder_path"]
                filename = dict_item["asset_file_name"]
                asset_options = dict_item["asset_options"]
                whole_body_semantic = dict_item["body_semantic_label"]
                per_link_semantic = dict_item["link_semantic_label"]
                semantic_masked_links = dict_item["semantic_masked_links"]
                semantic_id = dict_item["semantic_id"]
                color = dict_item["color"]
                collision_mask = dict_item["collision_mask"]

                loaded_asset = self.gym.load_asset(self.sim, folder_path, filename, asset_options)

                # debug assets options
                # def print_all_properties(obj):
                #     """
                #     打印类实例中所有 @property 的值。
                    
                #     参数:
                #     - obj: 类实例。
                #     """
                #     # 遍历 dir(obj)，过滤掉以双下划线开头的特殊属性和方法
                #     properties = [attr for attr in dir(obj) 
                #                 if not attr.startswith("__") and not callable(getattr(obj, attr))]
                    
                #     print("Properties and their values:")
                #     for prop in properties:
                #         try:
                #             value = getattr(obj, prop)  # 获取属性值
                #             print(f"{prop}: {value}")
                #         except Exception as e:
                #             print(f"{prop}: Error retrieving value ({e})")
                # print_all_properties(asset_options)

                assert not (whole_body_semantic and per_link_semantic)
                if semantic_id < 0:
                    object_segmentation_id = self.segmentation_counter
                    self.segmentation_counter += 1
                else:
                    object_segmentation_id = semantic_id

                asset_counter += 1

                env_asset_handle = self.gym.create_actor(env_handle, loaded_asset, start_pose, "env_asset_"+str(asset_counter), i, collision_mask, object_segmentation_id)
                self.env_asset_handles.append(env_asset_handle)
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
                        self.segmentation_counter += 1
                        self.gym.set_rigid_body_segmentation_id(env_handle, env_asset_handle, rb_index, self.segmentation_counter)
            
                if semantic_id != 4 and semantic_id != 5 and semantic_id != 8:
                    if color is None:
                        color = np.random.randint(low=50,high=200,size=3)

                    self.gym.set_rigid_body_color(env_handle, env_asset_handle, 0, gymapi.MESH_VISUAL,
                            gymapi.Vec3(color[0]/255,color[1]/255,color[2]/255))
        
        self.robot_body_items = self.gym.get_actor_rigid_body_properties(self.envs[0],self.actor_handles[0])
        self.robot_mass = 0
        for item in self.robot_body_items:
            self.robot_mass += item.mass
        print("Total robot mass: ", self.robot_mass)
        print("\n\n\n\n\n ENVIRONMENT CREATED \n\n\n\n\n\n")

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # if env_ids[0]:
        #     print("sssssssssssssssssssssssssssssssssssssssssssssssss")

        # reset cube position
        R = 2.0
        theta = torch.pi*torch_rand_float(-1.0, 1.0, (num_resets, 1), self.device)
        self.cube_states[env_ids, 0:1] = R*torch.cos(theta)
        self.cube_states[env_ids, 1:2] = R*torch.sin(theta)
        self.cube_states[env_ids, 2:3] = .0*torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1.
        # self.cube_states[env_ids, 0:2] = 0*torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device) + torch.tensor([1.5, 0.], device=self.device)
        # self.cube_states[env_ids, 2:3] = .0*torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1.
        self.cube_states[env_ids, 7:10] = self.calculate_cube_velocity(self.cube_positions, 5.0)

        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # randomize root states
        self.root_states[env_ids, 0:2] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device) + torch.tensor([0., 0.], device=self.device)
        self.root_states[env_ids, 2:3] = 0.*torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1.
        # self.root_states[env_ids, 0] = 0 # debug
        # self.root_states[env_ids, 1] = 0 # debug
        # self.root_states[env_ids, 2:3] = 1 # debug

        # randomize root orientation
        root_angle = torch.concatenate([0.*torch_rand_float(-torch.pi, torch.pi, (num_resets, 2), self.device), # .1
                                       0.*torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device)], dim=-1) # 0.2
        # root_angle = torch.concatenate([0.*torch.ones((num_resets, 1), device=self.device), # debug
        #                                 0.*torch.ones((num_resets, 1), device=self.device), # debug
        #                                 0.8*torch.pi*torch.ones((num_resets, 1), device=self.device)], dim=-1) # debug
        matrix = T.euler_angles_to_matrix(root_angle, 'XYZ')
        root_quats = T.matrix_to_quaternion(matrix) # w,x,y,z
        self.root_states[env_ids, 3:7] = root_quats[:, [1, 2, 3, 0]] #x,y,z,w

        # randomize root linear and angular velocities
        self.root_states[env_ids, 7:10] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device) # 0.5
        self.root_states[env_ids, 10:13] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device) # 0.2
        # self.root_states[env_ids, 7:10] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device) # debug
        # self.root_states[env_ids, 10:13] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device) # debug

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        
        self.reset_buf[env_ids] = 1
        self.progress_buf[env_ids] = 0

        self.thrust_cmds_damp[env_ids] = 0
        self.thrust_rot_damp[env_ids] = 0

        self.int_pos_error[env_ids] = 0
        self.int_yaw_error[env_ids] = 0

        self.pre_actions[env_ids] = 0
        self.pre_root_positions[env_ids] = 0

        if self.cfg.use_tcn:
            self.obs_seqs_buf[env_ids] = 0

    def calculate_cube_velocity(self, cube_position, v_e, g=9.81):
        """
        计算 cube 的初速度 (vx, vy, vz) 以使其以水平速度为基础，砸向无人机。

        参数：
        - cube_position (tensor): 初始位置 [num_envs, 3]，每行为 (x, y, z)。
        - v_e (float): 期望水平速度（标量）。
        - g (float): 重力加速度，默认 9.81。

        输出：
        - velocity (tensor): 初速度 [num_envs, 3]，每行为 (vx, vy, vz)。
        """
        # 无人机位置
        drone_position = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(cube_position)  # [num_envs, 3]

        # 计算方向向量
        direction = drone_position - cube_position  # [num_envs, 3]
        distance_xy = torch.norm(direction[:, :2], dim=1, keepdim=True)  # 水平距离 [num_envs, 1]
        unit_direction_xy = direction[:, :2] / distance_xy  # 水平方向单位向量 [num_envs, 2]

        # 计算水平飞行时间
        v_e = torch.tensor(v_e, device=self.device).expand_as(distance_xy)  # 期望水平速度 [num_envs, 1]
        t = distance_xy / v_e  # 水平运动时间 [num_envs, 1]

        # 计算垂直方向初速度
        z_c = cube_position[:, 2].unsqueeze(1)  # 初始高度 [num_envs, 1]
        z_u = drone_position[:, 2].unsqueeze(1)  # 无人机目标高度 [num_envs, 1]
        v_z = (z_u - z_c + 0.5 * g * t**2) / t  # 垂直速度公式 [num_envs, 1]

        # 水平方向初速度分量
        v_x = unit_direction_xy[:, 0].unsqueeze(1) * v_e  # 水平方向 x 分量 [num_envs, 1]
        v_y = unit_direction_xy[:, 1].unsqueeze(1) * v_e  # 水平方向 y 分量 [num_envs, 1]

        # 合并速度分量
        velocity = torch.cat([v_x, v_y, v_z], dim=1)  # [num_envs, 3]

        return velocity


    def compute_observations(self):
        self.root_matrix = T.quaternion_to_matrix(self.root_quats[:, [3, 0, 1, 2]]).reshape(self.num_envs, 9)
        # print(self.root_matrix)
        self.obs_buf[..., 0:9] = self.root_matrix
        self.obs_buf[..., 9:12] = self.root_positions
        self.obs_buf[..., 12:15] = self.root_linvels
        self.obs_buf[..., 15:18] = self.root_angvels

        # add position and velocity of cube
        self.cube_obs = torch.cat([self.cube_positions, self.cube_linvels], dim=-1)
        self.obs_buf[..., 18:24] -= self.cube_obs

        self.add_noise()

        # add cube noise
        cube_obs_noise = 5e-2 *torch_normal_float((self.num_envs, 6), self.device)
        self.obs_buf[..., 18:24] += cube_obs_noise
        return self.obs_buf

    def compute_reward(self):
        # print(self.root_quats)
        # print(self.pre_root_positions[0])
        self.rew_buf[:], self.reset_buf[:] ,self.item_reward_info = self.compute_quadcopter_reward(
            self.cube_positions,
            self.root_positions,
            self.pre_root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, 
            self.progress_buf, 
            self.max_episode_length, 
        )
        # print(self.actions[0])
        # update prev
        self.pre_actions = self.actions.clone()
        self.pre_root_positions = self.root_positions.clone()
        self.pre_root_angvels = self.root_angvels.clone()
    
    def hit_reward(self, root_positions, target_positions, progress_buf):
        check = torch.norm(target_positions-root_positions, dim=-1)
        hit_r = 800 * torch.where(check < 0.1, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))
        progress_r = 800 * torch.where(check < 0.1, 
                                        (1 - torch.clamp(progress_buf / self.max_episode_length, 0.0, 1.0)), 
                                        torch.tensor(0, device=self.device))
        return hit_r, progress_r, check
    
    def guidance_reward(self, root_positions, pre_root_positions, target_positions):
        if torch.all(pre_root_positions == 0):
            pre_root_positions = root_positions # avoid a high wrong reward value cause by the first root_positions
        r = 500 * (torch.norm(target_positions-pre_root_positions, dim=-1) - torch.norm(target_positions-root_positions, dim=-1))
        return r
    
    def continous_action_reward(self, root_angvels, pre_root_angvels, actions, pre_actions):
        angular_diff = root_angvels - pre_root_angvels
        max_angular_change = 1.0
        r1 = 0.4 * (1 - torch.clamp(angular_diff.norm(dim=1) / max_angular_change, 0.0, 1.0))
        action_diff = actions - pre_actions
        r2 = 0.4 * (1- torch.sqrt(action_diff[..., :-1].pow(2).sum(-1))/5) + 0.8 * (1-torch.sqrt(action_diff[..., -1].pow(2))/5)
        return r1+r2
    
    def vel_dir_reward(self, root_linvels, target_positions, root_positions):
        relative_positions = target_positions - root_positions
        tar_direction = relative_positions / torch.norm(relative_positions, dim=1, keepdim=True)
        vel_direction = root_linvels / torch.norm(root_linvels, dim=1, keepdim=True)
        dot_product = (tar_direction * vel_direction).sum(dim=1)
        angle_difference = torch.acos(dot_product.clamp(-1.0, 1.0)).abs()
        r = 3 * (1 - angle_difference / torch.pi)
        return r

    def compute_quadcopter_reward(self, target_positions, root_positions, pre_root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length):
        relative_positions = target_positions - root_positions
        
        guidance_reward = self.guidance_reward(root_positions, pre_root_positions, target_positions)
        hit_reward, progress_r, check = self.hit_reward(root_positions, target_positions, progress_buf)
        continous_action_reward = self.continous_action_reward(root_angvels, self.pre_root_angvels, self.actions, self.pre_actions)
        vel_dir_reward = self.vel_dir_reward(root_linvels, target_positions, root_positions)
        reward = guidance_reward + hit_reward + progress_r + continous_action_reward + vel_dir_reward

        # resets due to misbehavior
        ones = torch.ones_like(reset_buf)
        die = torch.zeros_like(reset_buf)

        # resets due to episode length
        reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

        # # reset if altitude is too low or too high
        # reset = torch.where(torch.logical_or(
        #     torch.logical_and(self.flag, self.root_positions[..., 2] > target_positions[..., 2]), 
        #     torch.logical_and(~self.flag, self.root_positions[..., 2] < target_positions[..., 2])), ones, reset)
        
        # # thrust must be clamp to -1 and 1
        # reset = torch.where(self.actions[..., -1] < -1, ones, reset)
        # reset = torch.where(self.actions[..., -1] > 1, ones, reset)

        # resets due to out of bounds
        reset = torch.where(torch.norm(relative_positions, dim=1) > 4, ones, reset)
        
        reset = torch.where(relative_positions[..., 2] < -2, ones, reset)
        reset = torch.where(relative_positions[..., 2] > 2, ones, reset)

        reset = torch.where(root_positions[..., 2] < 0, ones, reset)
        reset = torch.where(root_positions[..., 2] > 3, ones, reset)

        reset = torch.where(check < 0.1, ones, reset)
                
        item_reward_info = {}
        item_reward_info["guidance_reward"] = guidance_reward
        item_reward_info["hit_reward"] = hit_reward
        item_reward_info["progress_r"] = progress_r
        item_reward_info["continous_action_reward"] = continous_action_reward
        item_reward_info["vel_dir_reward"] = vel_dir_reward
        item_reward_info["reward"] = reward

        # print(guidance_reward[0], hit_reward[0], progress_r[0], continous_action_reward[0], vel_dir_reward[0], reward[0])

        return reward, reset, item_reward_info

###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

@torch.jit.script
def torch_normal_float(shape, device):
    # type: (Tuple[int, int], str) -> Tensor
    return torch.randn(*shape, device=device)