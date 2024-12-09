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
from airgym.envs.acrobatics.X152b_sigmoid_config import X152bSigmoidConfig

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

class X152bSigmoid(X152bPx4):

    def __init__(self, cfg: X152bSigmoidConfig, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        print("ctl mode =========== ",cfg.env.ctl_mode)
        self.ctl_mode = cfg.env.ctl_mode
        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.debug_viz = False
        num_actors = 1

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless

        super(X152bPx4, self).__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        bodies_per_env = self.robot_num_bodies

        self.vec_root_tensor = gymtorch.wrap_tensor(
            self.root_tensor).view(self.num_envs, num_actors, 13)

        self.root_states = self.vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[..., 0:3]
        self.root_quats = self.root_states[..., 3:7]
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
            # self.action_upper_limits = torch.ones((self.num_actions), device=self.device, dtype=torch.float32)
            # self.action_lower_limits = -torch.ones((self.num_actions), device=self.device, dtype=torch.float32)
            # self.action_lower_limits[-1] = 0.
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
        # parameters for the X152b
        self.target_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_positions[:, 2] = 1
        self.all_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        # control tensors
        self.action_input = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
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

        # control list and target flag
        self.control_lists_tensor = torch.tensor(self.cfg.env.control_lists).to(self.device)
        self.flag = torch.ones(self.num_envs, device=self.device)

        # reward integration buffers
        self.int_pos_error = torch.zeros((self.num_envs, 10), device=self.device)
        self.int_yaw_error = torch.zeros((self.num_envs, 10), device=self.device)

        self.pre_root_positions = torch.zeros((self.num_envs, 3), device=self.device)

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

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.root_states[env_ids] = self.initial_root_states[env_ids]

        self.root_states[env_ids, 0] = 1.0*torch_rand_float(.0, .0, (num_resets, 1), self.device).squeeze(-1) # 2.0
        self.root_states[env_ids, 1] = 1.0*torch_rand_float(-1.0, -1.0, (num_resets, 1), self.device).squeeze(-1) # 2.0
        self.root_states[env_ids, 2] = 1.0*torch_one_rand_float(1., 1., (num_resets, 1), self.device).squeeze(-1) # 2
        
        root_angle = torch.concatenate([0.*torch_rand_float(-torch.pi, torch.pi, (num_resets, 2), self.device), # .1
                                       0.*torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device)], dim=-1) # 0.2
        matrix = T.euler_angles_to_matrix(root_angle, 'XYZ')
        root_quats = T.matrix_to_quaternion(matrix) # w,x,y,z
        self.root_states[env_ids, 3:7] = root_quats[:, [1, 2, 3, 0]] #x,y,z,w

        self.root_states[env_ids, 7:10] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device) # 0.5
        self.root_states[env_ids, 10:13] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device) # 0.2

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        self.reset_buf[env_ids] = 1
        self.progress_buf[env_ids] = 0

        self.thrust_cmds_damp[env_ids] = 0
        self.thrust_rot_damp[env_ids] = 0

        self.int_pos_error[env_ids] = 0
        self.int_yaw_error[env_ids] = 0

        self.pre_actions[env_ids] = 0
        self.pre_root_positions[env_ids] = 0

        self.flag[env_ids] = 1

    def compute_observations(self):
        self.root_matrix = T.quaternion_to_matrix(self.root_quats[:, [3, 0, 1, 2]]).reshape(self.num_envs, 9)
        # print(self.root_matrix)
        self.obs_buf[..., 0:9] = self.root_matrix
        self.obs_buf[..., 9:12] = self.root_positions
        self.obs_buf[..., 12:15] = self.root_linvels
        self.obs_buf[..., 15:18] = self.root_angvels

        self.add_noise()
        return self.obs_buf

    def compute_reward(self):
        # print(self.root_quats)
        # print(self.pre_root_positions[0])
        self.rew_buf[:], self.reset_buf[:] ,self.item_reward_info = self.compute_quadcopter_reward(
            self.root_positions,
            self.pre_root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, 
            self.progress_buf, 
            self.max_episode_length, 
        )
        
        # update prev
        self.pre_actions = self.actions.clone()
        self.pre_root_positions = self.root_positions.clone()
    
    def guidance_reward(self, root_positions, root_linvels):
        cur_target = self.control_lists_tensor[self.flag.int()]
        # target = torch.tensor([3, 0, 0], device=self.device).repeat(self.num_envs, 1)
        tar_direction = cur_target - root_positions
        r = torch.sum(tar_direction * root_linvels, dim=-1)
        return r
    
    # def guidance_reward(self, root_positions, pre_root_positions, root_angvels):
    #     cur_target = self.control_lists_tensor[self.flag.int()]
    #     r = torch.norm(cur_target-pre_root_positions, dim=-1) - torch.norm(cur_target-root_positions, dim=-1) - 0.01 * torch.norm(root_angvels, dim=-1)
    #     return r

    def tracking_reward(self, root_positions, root_linvels):
        '''
        Three trajectories:
        x1 = 6*t
        y1 = -36*t**3 + 10*t - 1
        z1 = 1

        x2 = 6*t + 2
        y2 = 72*t**3 - 36*t**2 - 2*t + 1
        z2 = 1

        x3 = 6*t + 4
        y3 = -36*t**3 + 36*t**2 - 2*t - 1
        z3 = 1
        '''

        cond1 = root_positions[..., 0] < self.cfg.env.control_lists[1][0]
        cond2 = (root_positions[..., 0] >= self.cfg.env.control_lists[1][0]) & (root_positions[..., 0] < self.cfg.env.control_lists[2][0])
        cond3 = root_positions[..., 0] >= self.cfg.env.control_lists[2][0]

        t = torch.zeros_like(root_positions[..., 0])
        t = torch.where(cond1, root_positions[..., 0] / 6, t)
        t = torch.where(cond2, (root_positions[..., 0] - 2) / 6, t)
        t = torch.where(cond3, (root_positions[..., 0] - 4) / 6, t)

        #------------------------- position -------------------------#
        px1, py1, pz1 = 6 * t, -36 * t**3 + 10 * t - 1, torch.full_like(t, 1)
        px2, py2, pz2 = 6 * t + 2, 72 * t**3 - 36 * t**2 - 2 * t + 1, torch.full_like(t, 1)
        px3, py3, pz3 = 6 * t + 4, -36 * t**3 + 36 * t**2 - 2 * t - 1, torch.full_like(t, 1)

        # Combine the segments
        px = torch.where(cond1, px1, torch.where(cond2, px2, px3))
        py = torch.where(cond1, py1, torch.where(cond2, py2, py3))
        pz = torch.where(cond1, pz1, torch.where(cond2, pz2, pz3))

        # Combine into pose tensor
        pos = torch.stack((px, py, pz), dim=-1)
        pos_dist = torch.norm(root_positions - pos, dim=-1)
        pos_reward = 1.0 - pos_dist/6

        #------------------------- velocity -------------------------#
        vx1, vy1, vz1 = 6, -108 * t**2 + 10, torch.full_like(t, 0)
        vx2, vy2, vz2 = 6, 216 * t**2 - 72 * t - 2, torch.full_like(t, 0)
        vx3, vy3, vz3 = 6, -108 * t**2 + 72 * t - 2, torch.full_like(t, 0)

        # Combine the segments
        vx = torch.where(cond1, vx1, torch.where(cond2, vx2, vx3))
        vy = torch.where(cond1, vy1, torch.where(cond2, vy2, vy3))
        vz = torch.where(cond1, vz1, torch.where(cond2, vz2, vz3))

        # Combine into velocity tensor
        vel = torch.stack((vx, vy, vz), dim=-1)
        vel_dist = torch.norm(root_linvels - vel, dim=-1)
        vel_reward = 1 - vel_dist/6

        # print("pos_reward:", pos_reward[0], "vel_reward:", vel_reward[0])

        return 1.0 * pos_reward + 0.05 * vel_reward, pos_dist
    
    def update_flag(self):
        # check whether the agent has reached the current target. if yes, flag += 1 and turn to the next target
        cur_target = self.control_lists_tensor[self.flag.int()]
        self.flag = torch.where(torch.norm(self.root_positions - cur_target, dim=-1) < 0.2, self.flag+1, self.flag)
        
        # nxt_target = self.control_lists_tensor[self.flag.int()+1]
        # tar_vel = nxt_target - self.root_positions
        # cosin_theta = torch.sum(tar_vel * self.root_linvels, dim=-1) / (torch.norm(tar_vel, dim=-1) * torch.norm(self.root_linvels, dim=-1) + 1e-8)
        # cosin_theta = torch.clamp(cosin_theta, -1., 1.)
        # r = .05 * cosin_theta
        # return r

        return 0.1 * torch.where(torch.norm(self.root_positions - cur_target, dim=-1) < 0.2, 
                           torch.ones(self.num_envs, dtype=torch.float32, device=self.device), 
                           torch.zeros(self.num_envs, dtype=torch.float32, device=self.device))

    def compute_quadcopter_reward(self, root_positions, pre_root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length):
        
        reward, dist = self.tracking_reward(root_positions, root_linvels)

        # guidance_reward = self.guidance_reward(root_positions, pre_root_positions, root_angvels)
        # # print("guidance_reward:", guidance_reward[0])
        flag_reward = self.update_flag()
        # # print("flag_reward:", self.flag[0], flag_reward[0])
        # reward = guidance_reward

        # resets due to misbehavior
        ones = torch.ones_like(reset_buf)
        die = torch.zeros_like(reset_buf)
        # die = torch.where(target_dist > 10.0, ones, die)

        # resets due to episode length
        reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)
        # reset = torch.where(dist > 1.0, ones, reset)

        reset = torch.where(root_positions[..., 2] < 0.5, ones, reset)
        reset = torch.where(root_positions[..., 2] > 1.5, ones, reset)
        
        item_reward_info = {}
        # item_reward_info["guidance_reward"] = guidance_reward
        # item_reward_info["flag_reward"] = flag_reward

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