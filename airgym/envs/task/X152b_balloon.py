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
from airgym.envs.task.X152b_balloon_config import X152bBalloonConfig
from airgym.utils.asset_manager import AssetManager

from rlPx4Controller.pyParallelControl import ParallelRateControl,ParallelVelControl,ParallelAttiControl,ParallelPosControl

import matplotlib.pyplot as plt
from airgym.utils.helpers import asset_class_to_AssetOptions
from airgym.utils.rotations import quats_to_euler_angles
import time

import pytorch3d.transforms as T
import torch.nn.functional as F

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

def quaternion_norm(q):
    norm = torch.sqrt(q.pow(2).sum(dim=1, keepdim=True))
    return q / norm

class X152bBalloon(X152bPx4WithCam):

    def __init__(self, cfg: X152bBalloonConfig, sim_params, physics_engine, sim_device, headless):
        self.cam_resolution = cfg.env.cam_resolution # set camera resolution
        self.cam_channel = cfg.env.cam_channel # set camera channel
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cam_resolution = cfg.env.cam_resolution # recover camera resolution
        self.cam_channel = cfg.env.cam_channel # recover camera channel

        # get states of red balloon
        self.balloon_states = self.env_asset_root_states[:, 0, :]
        self.balloon_positions = self.balloon_states[..., 0:3]
        self.balloon_quats = self.balloon_states[..., 3:7] # x,y,z,w
        self.balloon_linvels = self.balloon_states[..., 7:10]
        self.balloon_angvels = self.balloon_states[..., 10:13]

        if self.cfg.env.enable_onboard_cameras:
            print("Onboard cameras enabled...")
            print("Checking camera resolution =========== ", self.cam_resolution)
            self.full_camera_array = torch.zeros((self.num_envs, self.cam_channel, self.cam_resolution[0], self.cam_resolution[1]), device=self.device) # 1 for depth
        self.full_camera_array = torch.zeros((self.num_envs, self.cam_channel, self.cam_resolution[0], self.cam_resolution[1]), device=self.device) # 1 for depth

        self.pre_root_positions = torch.zeros_like(self.root_positions)
        self.pre_root_linvels = torch.zeros_like(self.root_linvels)
        self.pre_root_angvels = torch.zeros_like(self.root_angvels)
        self.initial_root_pos = torch.zeros_like(self.root_positions)

        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env
            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        self.env_asset_manager.randomize_pose()
        self.env_asset_manager.specify_pose()

        # reset target red ball position
        self.balloon_states[env_ids, 0:1] = .5*torch_rand_float(-1.0, 1.0, (num_resets, 1), self.device) + torch.tensor([2.5], device=self.device)
        self.balloon_states[env_ids, 1:2] = 2.*torch_rand_float(-1.0, 1.0, (num_resets, 1), self.device) + torch.tensor([0.], device=self.device)
        self.balloon_states[env_ids, 2:3] = .3*torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1.
        # self.balloon_states[env_ids, 0:2] = 1.5*torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device) + torch.tensor([0, 0.], device=self.device)
        # self.balloon_states[env_ids, 2:3] = .5*torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1.

        # randomize root states
        self.root_states[env_ids, 0:2] = 0.1*torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device) + torch.tensor([0., 0.], device=self.device) # 0.1
        self.root_states[env_ids, 2:3] = 0.2*torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1. # 0.2
        # self.root_states[env_ids, 0] = 0 # debug
        # self.root_states[env_ids, 1] = 0 # debug
        # self.root_states[env_ids, 2:3] = 1 # debug

        # randomize root orientation
        """
        Note: randomed initial angle can encourage exploration at the beginning.
        """
        root_angle = torch.concatenate([0.1*torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device), # .1
                                        0.1*torch_rand_float(0., torch.pi, (num_resets, 1), self.device), # .1
                                       0.2*torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device)], dim=-1) # 0.2
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

        self.initial_root_pos[env_ids] = self.root_positions[env_ids, 0:3].clone()

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
        rate = 1 #self.cfg.env.cam_dt / self.cfg.sim.dt
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
        # obs = self.obs_buf

        return obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_observations(self):
        self.root_matrix = T.quaternion_to_matrix(self.root_quats[:, [3, 0, 1, 2]]).reshape(self.num_envs, 9)
        self.obs_buf[..., 0:9] = self.root_matrix
        self.obs_buf[..., 9:12] = self.root_positions
        self.obs_buf[..., 12:15] = self.root_linvels
        self.obs_buf[..., 15:18] = self.root_angvels
        self.add_noise()

        # add relative position of target ball
        self.balloon_matrix = T.quaternion_to_matrix(self.balloon_quats[:, [3, 0, 1, 2]]).reshape(self.num_envs, 9)
        self.obs_buf[..., 0:9] -= self.balloon_matrix
        self.obs_buf[..., 9:12] -= self.balloon_positions

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] ,self.item_reward_info = self.compute_quadcopter_reward()
        # update prev
        self.pre_actions = self.actions.clone()
        self.pre_root_positions = self.root_positions.clone()
        self.pre_root_angvels = self.root_angvels.clone()
    
    def hit_reward(self, root_positions, target_positions, progress_buf):
        check = torch.norm(target_positions-root_positions, dim=-1)
        hit_r = 800 * torch.where(check < 0.1, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))
        return hit_r, check

    def compute_quadcopter_reward(self):
        relative_positions = self.balloon_positions- self.root_positions

        direction_vector = F.normalize(relative_positions, dim=-1)
        direction_yaw = torch.atan2(direction_vector[..., 1], direction_vector[..., 0])
        root_matrix = T.quaternion_to_matrix(self.root_quats[:, [3, 0, 1, 2]])
        root_euler = T.matrix_to_euler_angles(root_matrix, convention='XYZ')
        relative_heading = compute_yaw_diff(root_euler[..., 2], direction_yaw)

        yaw_distance = torch.norm(relative_heading.unsqueeze(-1), dim=1)
        yaw_reward = 1.0 / (1.0 + torch.square(1.6 * yaw_distance)) 

        # # this rewarding causes high speed and high acceleration. If use this guidance reward
        # guidance_reward = 30 * (torch.norm(self.balloon_positions-self.pre_root_positions, dim=-1) - 
        #                                 torch.norm(self.balloon_positions-self.root_positions, dim=-1)) 
        initial_relative_positions = self.balloon_positions - self.initial_root_pos
        guidance_reward = 1 * torch.exp(-torch.norm(self.balloon_positions-self.root_positions, dim=-1) / torch.norm(initial_relative_positions, dim=-1))

        ups = quat_axis(self.root_quats, axis=2)
        ups_reward = 0.5 * torch.pow((ups[..., 2] + 1) / 2, 2)
        
        hit_reward, check = self.hit_reward(self.root_positions, self.balloon_positions, self.progress_buf)

        effort_reward = .1 * torch.exp(-self.actions.pow(2).sum(-1))
        action_diff = torch.norm(self.actions - self.pre_actions, dim=-1)
        action_smoothness_reward = .1 * torch.exp(-action_diff)

        reward = (
            guidance_reward
            + yaw_reward
            + hit_reward
            + action_smoothness_reward
            + ups_reward
            + effort_reward
        )

        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        # resets due to episode length
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, ones, die)
        
        # thrust must be clamp to -1 and 1
        reset = torch.where(self.actions[..., -1] < -1, ones, reset)
        reset = torch.where(self.actions[..., -1] > 1, ones, reset)

        # kill if far away from the target along x axis
        reset = torch.where(relative_positions[..., 0] < -0.2, ones, reset)

        reset = torch.where(self.root_linvels[..., 0] < 0, ones, reset)

        # resets due to out of bounds
        reset = torch.where(torch.norm(relative_positions, dim=1) > 4, ones, reset)

        reset = torch.where(self.root_positions[..., 2] < 0.5, ones, reset)
        reset = torch.where(self.root_positions[..., 2] > 1.5, ones, reset)

        reset = torch.where(check < 0.1, ones, reset)
                
        item_reward_info = {}
        item_reward_info["guidance_reward"] = guidance_reward
        item_reward_info["hit_reward"] = hit_reward
        item_reward_info["action_smoothness_reward"] = action_smoothness_reward
        item_reward_info["effort_reward"] = effort_reward
        item_reward_info["ups_reward"] = ups_reward
        item_reward_info["reward"] = reward

        return reward, reset, item_reward_info
        

###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_yaw_diff(a: torch.Tensor, b: torch.Tensor):
    """Compute the difference between two sets of Euler angles. a & b in [-pi, pi]"""
    diff = b - a
    diff = torch.where(diff < -torch.pi, diff + 2*torch.pi, diff)
    diff = torch.where(diff > torch.pi, diff - 2*torch.pi, diff)
    return diff

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