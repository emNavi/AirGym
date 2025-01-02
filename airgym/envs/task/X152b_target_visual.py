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
from airgym.envs.task.X152b_target_visual_config import X152bTargetVisualConfig
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

class X152bTargetVisual(X152bPx4WithCam):

    def __init__(self, cfg: X152bTargetVisualConfig, sim_params, physics_engine, sim_device, headless):
        self.cam_resolution = cfg.env.cam_resolution # set camera resolution
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cam_resolution = cfg.env.cam_resolution # recover camera resolution

        # get states of red balloon
        self.target_ball_states = self.env_asset_root_states[:, 0, :]
        self.target_ball_positions = self.target_ball_states[..., 0:3]
        self.target_ball_quats = self.target_ball_states[..., 3:7] # x,y,z,w
        self.target_ball_linvels = self.target_ball_states[..., 7:10]
        self.target_ball_angvels = self.target_ball_states[..., 10:13]

        if self.cfg.env.enable_onboard_cameras:
            print("Onboard cameras enabled...")
            print("Checking camera resolution =========== ", self.cam_resolution)
            self.full_camera_array = torch.zeros((self.num_envs, 1, self.cam_resolution[0], self.cam_resolution[1]), device=self.device) # 1 for depth

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
        
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        self.env_asset_manager.randomize_pose()

        # reset target red ball position
        self.target_ball_states[env_ids, 0:1] = .5*torch_rand_float(-1.0, 1.0, (num_resets, 1), self.device) + torch.tensor([1.5], device=self.device)
        self.target_ball_states[env_ids, 1:2] = 1.*torch_rand_float(-1.0, 1.0, (num_resets, 1), self.device) + torch.tensor([0.], device=self.device)
        self.target_ball_states[env_ids, 2:3] = .3*torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1.
        # self.target_ball_states[env_ids, 0:2] = 0*torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device) + torch.tensor([1.5, 0.], device=self.device)
        # self.target_ball_states[env_ids, 2:3] = .0*torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1.

        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # randomize root states
        self.root_states[env_ids, 0:2] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device) + torch.tensor([0., 0.], device=self.device)
        self.root_states[env_ids, 2:3] = 0.2*torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1.
        # self.root_states[env_ids, 0] = 0 # debug
        # self.root_states[env_ids, 1] = 0 # debug
        # self.root_states[env_ids, 2:3] = 1 # debug

        # randomize root orientation
        root_angle = torch.concatenate([0.01*torch_rand_float(-torch.pi, torch.pi, (num_resets, 2), self.device), # .1
                                       0.01*torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device)], dim=-1) # 0.2
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
        # step physics and render each frame
        for i in range(self.cfg.env.num_control_steps_per_env_step):
            self.pre_physics_step(actions)
            self.gym.simulate(self.sim)
            # NOTE: as per the isaacgym docs, self.gym.fetch_results must be called after self.gym.simulate, but not having it here seems to work fine
            # it is called in the render function.
            self.post_physics_step()
            self.progress_buf += 1

        self.render(sync_frame_time=False)
        if self.enable_onboard_cameras:
            self.render_cameras()
        
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

        obs = {
            'state': self.obs_buf,
            'image': self.full_camera_array,
        }
        return obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_observations(self):
        self.root_matrix = T.quaternion_to_matrix(self.root_quats[:, [3, 0, 1, 2]]).reshape(self.num_envs, 9)
        self.obs_buf[..., 0:9] = self.root_matrix
        self.obs_buf[..., 9:12] = self.root_positions
        self.obs_buf[..., 12:15] = self.root_linvels
        self.obs_buf[..., 15:18] = self.root_angvels
        self.add_noise()

        self.obs_buf[..., 18:22] = self.actions
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

    def compute_quadcopter_reward(self):
        relative_positions = self.target_ball_positions - self.root_positions
        
        guidance_reward = self.guidance_reward(self.root_positions, self.pre_root_positions, self.target_ball_positions)
        hit_reward, progress_r, check = self.hit_reward(self.root_positions, self.target_ball_positions, self.progress_buf)
        continous_action_reward = self.continous_action_reward(self.root_angvels, self.pre_root_angvels, self.actions, self.pre_actions)
        vel_dir_reward = self.vel_dir_reward(self.root_linvels, self.target_ball_positions, self.root_positions)
        
        reward = (
            guidance_reward
            + hit_reward
            + progress_r
            + continous_action_reward
            + vel_dir_reward
        )

        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        # resets due to episode length
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, ones, die)

        # # reset if altitude is too low or too high
        # reset = torch.where(torch.logical_or(
        #     torch.logical_and(self.flag, self.root_positions[..., 2] > target_positions[..., 2]), 
        #     torch.logical_and(~self.flag, self.root_positions[..., 2] < target_positions[..., 2])), ones, reset)
        
        # # thrust must be clamp to -1 and 1
        # reset = torch.where(self.actions[..., -1] < -1, ones, reset)
        # reset = torch.where(self.actions[..., -1] > 1, ones, reset)

        # resets due to out of bounds
        reset = torch.where(torch.norm(relative_positions, dim=1) > 4, ones, reset)

        reset = torch.where(self.root_positions[..., 2] < 0.5, ones, reset)
        reset = torch.where(self.root_positions[..., 2] > 1.5, ones, reset)

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