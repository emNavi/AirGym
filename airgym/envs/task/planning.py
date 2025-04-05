import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET

from isaacgym import gymutil, gymtorch, gymapi
from airgym import AIRGYM_ROOT_DIR, AIRGYM_ROOT_DIR
from airgym.utils.torch_utils import *
from airgym.envs.base.customized import Customized
import airgym.utils.rotations as rot_utils
from airgym.envs.task.planning_config import PlanningConfig
from airgym.assets.asset_manager import AssetManager

from rlPx4Controller.pyParallelControl import ParallelRateControl,ParallelVelControl,ParallelAttiControl,ParallelPosControl

import matplotlib.pyplot as plt
from airgym.utils.helpers import asset_class_to_AssetOptions
from airgym.utils.rotations import quats_to_euler_angles
import time

import pytorch3d.transforms as T
import torch.nn.functional as F

LENGTH = 8.0
WIDTH = 4.0
FLY_HEIGHT = 1.0

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

class Planning(Customized):

    def __init__(self, cfg: PlanningConfig, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # get states of goal
        self.goal_states = self.env_asset_root_states[:, 0, :]
        self.goal_positions = self.goal_states[..., 0:3]
        self.goal_quats = self.goal_states[..., 3:7] # x,y,z,w
        self.goal_linvels = self.goal_states[..., 7:10]
        self.goal_angvels = self.goal_states[..., 10:13]

        self.pre_root_positions = torch.zeros_like(self.root_positions)
        self.pre_root_linvels = torch.zeros_like(self.root_linvels)
        self.pre_root_angvels = torch.zeros_like(self.root_angvels)
        self.prev_related_dist = torch.zeros((self.num_envs), device=self.device)

        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env
            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize asset root states
        self.env_asset_root_states[env_ids, :, 0:1] = LENGTH * torch_rand_float(-1.0, 1.0, (num_resets, self.num_assets, 1), self.device) + torch.tensor([0.], device=self.device)
        self.env_asset_root_states[env_ids, :, 1:2] = WIDTH * torch_rand_float(-1.0, 1.0, (num_resets, self.num_assets, 1), self.device) + torch.tensor([0.], device=self.device)
        self.env_asset_root_states[env_ids, :, 2:3] = 0
        assets_root_angle = torch.concatenate([0 * torch_rand_float(-torch.pi, torch.pi, (num_resets, self.num_assets, 2), self.device),
                                       torch_rand_float(-torch.pi, torch.pi, (num_resets, self.num_assets, 1), self.device)], dim=-1)
        assets_matrix = T.euler_angles_to_matrix(assets_root_angle, 'XYZ')
        assets_root_quats = T.matrix_to_quaternion(assets_matrix)
        self.env_asset_root_states[env_ids, :, 3:7] = assets_root_quats[:, :, [1, 2, 3, 0]]

        # randomize goal states
        self.goal_states[env_ids, 0:1] = torch.tensor([LENGTH+0.5], device=self.device)
        self.goal_states[env_ids, 1:2] = 1.5 * torch_rand_float(-1.0, 1.0, (num_resets, 1), self.device) + torch.tensor([0.], device=self.device)
        self.goal_states[env_ids, 2:3] = .0 * torch_rand_float(-1., 1., (num_resets, 1), self.device) + FLY_HEIGHT
        # self.goal_states[env_ids, 0:2] = 0 * torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device) + torch.tensor([0., -5.], device=self.device) # debug
        # self.goal_states[env_ids, 2:3] = 0 * torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1. # debug

        # randomize root states
        self.root_states[env_ids, 0:2] = torch.tensor([-LENGTH-0.5, 0.], device=self.device)
        self.root_states[env_ids, 2:3] = .0 *torch_rand_float(-1., 1., (num_resets, 1), self.device) + FLY_HEIGHT
        # self.root_states[env_ids, 0:2] = 0 *torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device) + torch.tensor([-5., 0.], device=self.device) # debug
        # self.root_states[env_ids, 2:3] = 0 *torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1. # debug

        def compute_direction_angle(a, b, degrees=True):
            vector = b - a
            dx = vector[..., 0]
            dy = vector[..., 1]
            radians = torch.atan2(dy, dx)
            
            if degrees:
                angles = torch.rad2deg(radians)
            else:
                angles = radians
            
            return angles

        init_yaw = compute_direction_angle(self.root_states[env_ids, 0:2], self.goal_states[env_ids, 0:2], degrees=False).unsqueeze(-1)

        # randomize root orientation
        root_angle = torch.concatenate([0.*torch_rand_float(-torch.pi, torch.pi, (num_resets, 2), self.device), # .01
                                        0.*torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device) + init_yaw], dim=-1) # 0.05
        # root_angle = torch.concatenate([0.*torch_rand_float(-torch.pi, torch.pi, (num_resets, 2), self.device),# debug
        #                                 0.*torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device) + init_yaw], dim=-1) # debug

        matrix = T.euler_angles_to_matrix(root_angle, 'XYZ')
        root_quats = T.matrix_to_quaternion(matrix) # w,x,y,z
        self.root_states[env_ids, 3:7] = root_quats[:, [1, 2, 3, 0]] #x,y,z,w

        # randomize root linear and angular velocities
        self.root_states[env_ids, 7:10] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device) # 0.5
        self.root_states[env_ids, 10:13] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device) # 0.2

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)

        self.reset_buf[env_ids] = 1
        self.progress_buf[env_ids] = 0

        self.pre_actions[env_ids] = 0
        self.prev_related_dist[env_ids] = 0
        self.pre_root_positions[env_ids] = 0
        self.pre_root_angvels[env_ids] = 0

        q_global = self.root_quats[:, [3, 0, 1, 2]]
        rot_matrix_global = T.quaternion_to_matrix(q_global)  # (num_envs, 3, 3)

        yaw = torch.atan2(rot_matrix_global[:, 1, 0], rot_matrix_global[:, 0, 0])  # 计算 yaw (num_envs,)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        self.world_to_local = torch.stack([
            torch.stack([cos_yaw, -sin_yaw, torch.zeros_like(yaw)], dim=1),
            torch.stack([sin_yaw,  cos_yaw, torch.zeros_like(yaw)], dim=1),
            torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
        ], dim=2)

        self.esdf_dist = torch.ones((self.num_envs, 1), device=self.device) * 10
        
    def step(self, actions):
        """
        step physics and render each frame. 
        """
        # print("actions: ", actions)
        self.actions_local = actions.to(self.device)
        
        for i in range(self.cfg.env.num_control_steps_per_env_step):
            self.pre_physics_step(actions)
            self.gym.simulate(self.sim)
            # NOTE: as per the isaacgym docs, self.gym.fetch_results must be called after self.gym.simulate, but not having it here seems to work fine
            # it is called in the render function.
            self.post_physics_step()

        self.render(sync_frame_time=False)
        rate = self.cfg.env.cam_dt / self.cfg.sim.dt
        if self.counter % rate == 0:
            if self.enable_onboard_cameras:
                self.render_cameras()

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
        # 对每一张深度图计算最小深度值
        flattened_array = self.full_camera_array.clone().view(self.full_camera_array.size(0), -1)
        self.esdf_dist = torch.min(flattened_array, dim=1, keepdim=False).values

        # obs = self.obs_buf

        self.prev_related_dist = self.related_dist
        return obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_observations(self):
        forward_global = self.goal_positions - self.root_positions
        
        q_global = self.root_quats[:, [3, 0, 1, 2]]
        rot_matrix_global = T.quaternion_to_matrix(q_global)  # (num_envs, 3, 3)

        yaw = torch.atan2(rot_matrix_global[:, 1, 0], rot_matrix_global[:, 0, 0])  # 计算 yaw (num_envs,)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        self.world_to_local = torch.stack([
            torch.stack([cos_yaw, -sin_yaw, torch.zeros_like(yaw)], dim=1),
            torch.stack([sin_yaw,  cos_yaw, torch.zeros_like(yaw)], dim=1),
            torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
        ], dim=2)

        rot_matrix_local = torch.bmm(self.world_to_local, rot_matrix_global)
        self.euler_angles_local = T.matrix_to_euler_angles(rot_matrix_local, "XYZ")

        self.pos_diff_local = torch.einsum("bij,bj->bi", self.world_to_local, forward_global)
        self.vel_local = torch.einsum("bij,bj->bi", self.world_to_local, self.root_linvels)
        self.ang_vel_local = torch.einsum("bij,bj->bi", self.world_to_local, self.root_angvels)
        
        # self.obs_buf[..., 0:3] = self.pos_diff_local
        # self.obs_buf[..., 3] = self.related_dist = torch.norm(forward_global, dim=-1)
        # self.obs_buf[..., 4:7] = self.euler_angles_local
        # self.obs_buf[..., 7:10] = self.vel_local
        # self.obs_buf[..., 10:13] = self.ang_vel_local
        # self.obs_buf[..., 13:17] = self.actions_local

        self.goal_dir = self.pos_diff_local / torch.norm(self.pos_diff_local, dim=-1, keepdim=True)
        self.related_dist = torch.norm(forward_global, dim=-1)
        self.obs_buf[..., 0:3] = self.goal_dir
        self.obs_buf[..., 3:6] = self.euler_angles_local
        self.obs_buf[..., 6:9] = self.vel_local
        self.obs_buf[..., 9:12] = self.ang_vel_local
        self.obs_buf[..., 12:16] = self.actions_local

        # self.add_noise()
        
    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] ,self.item_reward_info = self.compute_quadcopter_reward()
        # update prev
        self.pre_actions = self.actions.clone()
        self.pre_root_positions = self.root_positions.clone()
        self.pre_root_angvels = self.root_angvels.clone()

    def compute_quadcopter_reward(self):
        # continous actions
        action_diff = self.actions - self.pre_actions
        continous_action_reward =  .2 * torch.norm(self.ang_vel_local, dim=-1) + .2 * torch.norm(action_diff, dim=-1)
        thrust_reward = .5 * (1-torch.abs(0.1533 - self.actions[..., -1]))
        
        # guidance reward
        forward_reward = 5 * (torch.norm(self.goal_positions - self.pre_root_positions, dim=-1) - torch.norm(self.goal_positions - self.root_positions, dim=-1))

        # heading reward
        forward_vec = self.pos_diff_local / torch.norm(self.pos_diff_local, dim=-1, keepdim=True)
        heading_vec = torch.tensor([1.0, 0.0, 0.0]).repeat(self.num_envs, 1).to(self.device)
        heading_reward = torch.sum(forward_vec * heading_vec, dim=-1)

        # speed reward
        speed_reward = torch.max(1 - torch.exp(torch.max(torch.tensor(0.0), torch.norm(self.vel_local, dim=-1) - 1.5)), torch.tensor(-1.0))

        # height reward
        z_reward = torch.min(torch.min(self.root_positions[..., 2] - 1.8, torch.tensor(0.0)), 1.2 - self.root_positions[..., 2])

        # ups reward
        ups = quat_axis(self.root_quats, axis=2)
        ups_reward = torch.square((ups[..., 2] + 1) / 2)

        # esdf reward
        esdf_reward = 0.5 * (1-torch.exp(- 0.5 * torch.square(self.esdf_dist))).squeeze(-1)

        # collision
        # alive_reward = torch.where(self.collisions > 0, -10., 0)
        alive_reward = torch.where(self.esdf_dist > 0.3, torch.tensor(0.0), torch.tensor(-10.0)).squeeze(-1)

        # reach goal
        reach_goal = self.related_dist < 0.3
        reach_goal_reward = torch.where(reach_goal, torch.tensor(0.0), torch.tensor(0.0))

        reward = (
            continous_action_reward
            + forward_reward
            + alive_reward
            + ups_reward
            + z_reward
            + esdf_reward
            + speed_reward
            + heading_reward
            + thrust_reward
            + reach_goal_reward
        )

        # print(continous_action_reward.shape)
        # print(forward_reward.shape)
        # print(alive_reward.shape)
        # print(z_reward.shape)
        # print(esdf_reward.shape)
        # print(speed_reward.shape)
        # print(heading_reward.shape)
        # print(thrust_reward.shape)
        # print(reach_goal_reward.shape)

        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        # resets due to too low or too high
        reset = torch.where(self.root_positions[..., 2] < FLY_HEIGHT-0.3, ones, die)
        reset = torch.where(self.root_positions[..., 2] > FLY_HEIGHT+0.3, ones, reset)

        # resets out off bound
        reset = torch.where(self.root_positions[..., 0] < -LENGTH-0.5, ones, reset)
        reset = torch.where(self.root_positions[..., 0] > LENGTH+0.5, ones, reset)
        reset = torch.where(self.root_positions[..., 1] < -WIDTH, ones, reset)
        reset = torch.where(self.root_positions[..., 1] > WIDTH, ones, reset)

        # resets due to collision or reach goal
        reset = torch.where(self.collisions > 0, ones, reset)
        reset = torch.where(reach_goal, ones, reset)

        # resets
        reset = torch.where(heading_reward < 0.25, ones, reset)

        # resets due to episode length
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, ones, reset)

        item_reward_info = {}
        item_reward_info["continous_action_reward"] = continous_action_reward
        item_reward_info["heading_reward"] = heading_reward
        item_reward_info["speed_reward"] = speed_reward
        item_reward_info["forward_reward"] = forward_reward
        item_reward_info["alive_reward"] = alive_reward
        item_reward_info["ups_reward"] = ups_reward
        item_reward_info["z_reward"] = z_reward
        item_reward_info["esdf_reward"] = esdf_reward
        item_reward_info["thrust_reward"] = thrust_reward
        item_reward_info["reach_goal_reward"] = reach_goal_reward
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