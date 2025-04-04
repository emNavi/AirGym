import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET

from airgym import AIRGYM_ROOT_DIR, AIRGYM_ROOT_DIR

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from airgym.envs.base.X152bPx4_with_cam import X152bPx4WithCam
import airgym.utils.rotations as rot_utils
from airgym.envs.task.X152b_avoid_config import X152bAvoidConfig
from airgym.assets.asset_manager import AssetManager

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

def compute_yaw_diff(a: torch.Tensor, b: torch.Tensor):
    """Compute the difference between two sets of Euler angles. a & b in [-pi, pi]"""
    diff = b - a
    diff = torch.where(diff < -torch.pi, diff + 2*torch.pi, diff)
    diff = torch.where(diff > torch.pi, diff - 2*torch.pi, diff)
    return diff

class X152bAvoid(X152bPx4WithCam):

    def __init__(self, cfg: X152bAvoidConfig, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # get states of object
        self.object_states = self.env_asset_root_states[:, 0, :]
        self.object_positions = self.object_states[..., 0:3]
        self.object_quats = self.object_states[..., 3:7] # x,y,z,w
        self.object_linvels = self.object_states[..., 7:10]
        self.object_angvels = self.object_states[..., 10:13]

        self.pre_root_positions = torch.zeros_like(self.root_positions)
        self.pre_root_linvels = torch.zeros_like(self.root_linvels)
        self.pre_root_angvels = torch.zeros_like(self.root_angvels)

        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env
            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def calculate_object_velocity(self, object_position, v_e, g=9.81):
        """
        计算 object 的初速度 (vx, vy, vz) 以使其以水平速度为基础，砸向无人机。
        
        参数：
        - object_position (tensor): 初始位置 [num_envs, 3]
        - v_e (float): 期望水平速度（标量）
        - g (float): 重力加速度，默认 9.81
        """
        # 确保输入维度正确
        if len(object_position.shape) == 3:
            object_position = object_position.squeeze(1)  # 如果是 [num_envs, 1, 3] 转换为 [num_envs, 3]
        
        # 过滤掉固定位置的cases
        valid_positions = (object_position[:, 0] != -999)
        if not torch.any(valid_positions):
            return torch.zeros_like(object_position)
            
        # 只处理有效的位置
        positions = object_position[valid_positions]
        num_valid = positions.shape[0]
        
        # 生成随机目标点
        drone_position = 0.3 * torch_rand_float(-1.0, 1.0, (num_valid, 3), self.device) + \
                        torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # 计算方向向量
        direction = drone_position - positions
        distance_xy = torch.norm(direction[:, :2], dim=1, keepdim=True)
        unit_direction_xy = direction[:, :2] / distance_xy

        # 计算水平飞行时间
        v_e = torch.tensor(v_e, device=self.device).expand_as(distance_xy)
        t = distance_xy / v_e

        # 计算垂直方向初速度
        z_c = positions[:, 2].unsqueeze(1)
        z_u = drone_position[:, 2].unsqueeze(1)
        v_z = (z_u - z_c + 0.5 * g * t**2) / t

        # 水平方向初速度分量
        v_x = unit_direction_xy[:, 0].unsqueeze(1) * v_e
        v_y = unit_direction_xy[:, 1].unsqueeze(1) * v_e

        # 合并速度分量
        velocity = torch.zeros_like(object_position)
        velocity[valid_positions] = torch.cat([v_x, v_y, v_z], dim=1)

        return velocity
        
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # Generate random values with correct shape (num_resets, 1)
        random_values = torch_rand_float(0.0, 1.0, (num_resets, 1), self.device)
        
        # Create masks for different initialization strategies
        random_pos_mask = random_values < 0.8  # 80% probability
        fixed_pos_mask = ~random_pos_mask      # 20% probability
        
        # Handle random position cases (80% probability)
        if torch.any(random_pos_mask):
            random_env_ids = env_ids[random_pos_mask.squeeze()]
            random_num_resets = random_pos_mask.sum().item()
            
            # Original initialization logic
            R = 4.2
            theta = torch.pi/6 * torch_rand_float(-1.0, 1.0, (random_num_resets, 1), self.device)
            
            self.object_states[random_env_ids, 0:1] = R * torch.cos(theta)
            self.object_states[random_env_ids, 1:2] = R * torch.sin(theta)
            self.object_states[random_env_ids, 2:3] = 0.0 * torch_rand_float(-1., 1., (random_num_resets, 1), self.device) + 1.4
            
            positions = self.object_states[random_env_ids, 0:3].squeeze(1)  # 确保维度正确
            velocities = self.calculate_object_velocity(positions, 4.5)
            self.object_states[random_env_ids, 7:10] = velocities
        
        # Handle fixed position cases (20% probability)
        if torch.any(fixed_pos_mask):
            fixed_env_ids = env_ids[fixed_pos_mask.squeeze()]
            
            # Set fixed position (-999, -999, 0)
            self.object_states[fixed_env_ids, 0:3] = torch.tensor([-999., -999., 0.], device=self.device)
            # Set zero velocity
            self.object_states[fixed_env_ids, 7:10] = 0.0

        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # randomize root states
        self.root_states[env_ids, 0:2] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device) + torch.tensor([0., 0.], device=self.device)
        self.root_states[env_ids, 2:3] = 0.2*torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1.
        # self.root_states[env_ids, 0] = 0 # debug
        # self.root_states[env_ids, 1] = 0 # debug
        # self.root_states[env_ids, 2:3] = 1 # debug

        # randomize root orientation
        root_angle = torch.concatenate([0.01*torch_rand_float(-torch.pi, torch.pi, (num_resets, 2), self.device), # .01
                                       0.05*torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device)], dim=-1) # 0.05
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

        self.pre_actions[env_ids] = 0
        self.pre_root_positions[env_ids] = 0
        self.pre_root_angvels[env_ids] = 0

    def step(self, actions):
        # print("actions: ", actions)
        # step physics and render each frame
        for i in range(self.cfg.env.num_control_steps_per_env_step):
            self.pre_physics_step(actions)
            self.gym.simulate(self.sim)
            # NOTE: as per the isaacgym docs, self.gym.fetch_results must be called after self.gym.simulate, but not having it here seems to work fine
            # it is called in the render function.
            self.post_physics_step()

        self.render(sync_frame_time=False)
        rate = self.cfg.env.cam_dt / self.cfg.sim.dt
        # print(self.counter)
        if self.counter % rate == 0:
            if self.enable_onboard_cameras:
                self.render_cameras()
        # print(self.full_camera_array[0], self.obs_buf[0])

        self.progress_buf += 1
        self.check_collisions()
        self.compute_observations()
        self.compute_reward()

        if self.cfg.env.reset_on_collision:
            ones = torch.ones_like(self.reset_buf)
            self.reset_buf = torch.where(self.collisions > 0, ones, self.reset_buf)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.time_out_buf = self.progress_buf > self.max_episode_length
        self.extras["time_outs"] = self.time_out_buf
        self.extras["item_reward_info"] = self.item_reward_info

        obs = {
            'image': self.full_camera_array,
            'observation': self.obs_buf,
        }

        return obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_observations(self):
        self.root_matrix = T.quaternion_to_matrix(self.root_quats[:, [3, 0, 1, 2]]).reshape(self.num_envs, 9)
        self.obs_buf[..., 0:9] = self.root_matrix
        self.obs_buf[..., 9:12] = self.root_positions
        self.obs_buf[..., 12:15] = self.root_linvels
        self.obs_buf[..., 15:18] = self.root_angvels
        self.add_noise()

        self.obs_buf[..., 0:18] -= self.target_states
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] ,self.item_reward_info = self.compute_quadcopter_reward()
        # update prev
        self.pre_actions = self.actions.clone()
        self.pre_root_positions = self.root_positions.clone()
        self.pre_root_angvels = self.root_angvels.clone()

    def compute_quadcopter_reward(self):
        target_positions = self.target_states[..., 9:12]
        relative_positions = target_positions - self.root_positions

        target_matrix = self.target_states[..., 0:9].reshape(self.num_envs, 3,3)
        target_euler = T.matrix_to_euler_angles(target_matrix, 'XYZ')
        root_matrix = T.quaternion_to_matrix(self.root_quats[:, [3, 0, 1, 2]])
        root_euler = T.matrix_to_euler_angles(root_matrix, convention='XYZ')
        relative_heading = compute_yaw_diff(target_euler[..., 2], root_euler[..., 2])

        distance = torch.norm(torch.cat((relative_positions, relative_heading.unsqueeze(-1)), dim=-1), dim=1)
        pose_reward = 1.0 / (1.0 + torch.square(1.6 * distance))
        
        ups = quat_axis(self.root_quats, axis=2)
        ups_reward = torch.square((ups[..., 2] + 1) / 2)

        spinnage = torch.square(self.root_angvels[:, -1])
        spin_reward = 1.0 / (1.0 + torch.square(spinnage))

        effort_reward = .1 * torch.exp(-self.actions.pow(2).sum(-1))
        action_diff = torch.norm(self.actions[..., :-1] - self.pre_actions[..., :-1], dim=-1)
        # action_diff = torch.norm(self.actions - self.pre_actions, dim=-1)
        thrust_reward = .05 * (1-torch.abs(0.1533 - self.actions[..., -1]))
        action_smoothness_reward = .1 * torch.exp(-action_diff)

        alive_reward = torch.where(self.collisions > 0, -500., 0.5)

        assert pose_reward.shape == ups_reward.shape == spin_reward.shape
        reward = (
            pose_reward
            + pose_reward * (ups_reward + spin_reward)
            + effort_reward
            + action_smoothness_reward
            + thrust_reward
            + alive_reward
        )

        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        # resets due to episode length
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, ones, die)
        
        reset = torch.where(self.root_positions[..., 2] < .3, ones, reset)
        reset = torch.where(self.root_positions[..., 2] > 1.7, ones, reset)

        reset = torch.where(relative_positions.norm(dim=-1) > 2.0, ones, reset)

        reset = torch.where(ups[..., 2] < 0.0, ones, reset) # orient_z 小于0 = 飞行器朝下了

        item_reward_info = {}
        item_reward_info["pose_reward"] = pose_reward
        item_reward_info["ups_reward"] = ups_reward
        item_reward_info["spin_reward"] = spin_reward
        item_reward_info["effort_reward"] = effort_reward
        item_reward_info["action_smoothness_reward"] = action_smoothness_reward
        item_reward_info["thrust_reward"] = thrust_reward
        item_reward_info["alive_reward"] = alive_reward
        item_reward_info["reward"] = reward

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
