import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET


from airgym import AIRGYM_ROOT_DIR, AIRGYM_ROOT_DIR

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from airgym.envs.base.base_task import BaseTask
import airgym.utils.rotations as rot_utils
from airgym.envs.base.X152bPx4_config import X152bPx4Cfg

from pyParallelControl import ParallelRateControl,ParallelVelControl

import matplotlib.pyplot as plt
from airgym.utils.helpers import asset_class_to_AssetOptions
import time


class X152bPx4(BaseTask):

    def __init__(self, cfg: X152bPx4Cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg

        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.debug_viz = False
        num_actors = 1

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
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

        self.action_upper_limits = torch.tensor(
            [1, 1, 1, 1], device=self.device, dtype=torch.float32)
        self.action_lower_limits = torch.tensor(
            [-1, -1, -1, -1], device=self.device, dtype=torch.float32)
        
        # controller
        self.cmd_thrusts = torch.zeros((self.num_envs, 4))
        # choice 1 from rate ctrl and vel ctrl
        self.parallel_rate_control = ParallelRateControl(self.num_envs)
        self.parallel_vel_control = ParallelVelControl(self.num_envs)

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

        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env
            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_envs()
        self.progress_buf = torch.zeros(
            self.cfg.env.num_envs, device=self.sim_device, dtype=torch.long)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self):
        print("\n\n\n\n\n CREATING ENVIRONMENT \n\n\n\n\n\n")
        asset_path = self.cfg.robot_asset.file.format(
            AIRGYM_ROOT_DIR=AIRGYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = asset_class_to_AssetOptions(self.cfg.robot_asset)

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)

        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        start_pose = gymapi.Transform()
        self.env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-self.env_spacing, -
                                self.env_spacing, -self.env_spacing)
        env_upper = gymapi.Vec3(
            self.env_spacing, self.env_spacing, self.env_spacing)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = torch.tensor([0, 0, 1], device=self.device)
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.robot_asset.name, i, self.cfg.robot_asset.collision_mask, 0)
            
            self.robot_bodies = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        
        self.robot_mass = 0
        for body in self.robot_bodies:
            self.robot_mass += body.mass
        print("Total robot mass: ", self.robot_mass)
        
        print("\n\n\n\n\n ENVIRONMENT CREATED \n\n\n\n\n\n")

    def step(self, actions):
        # step physics and render each frame
        for i in range(self.cfg.env.num_control_steps_per_env_step):
            self.pre_physics_step(actions)
            self.gym.simulate(self.sim)
            # NOTE: as per the isaacgym docs, self.gym.fetch_results must be called after self.gym.simulate, but not having it here seems to work fine
            # it is called in the render function.
            self.post_physics_step()

        self.render(sync_frame_time=False)
        
        self.progress_buf += 1
        self.compute_observations()
        self.compute_reward()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.time_out_buf = self.progress_buf > self.max_episode_length
        self.extras["time_outs"] = self.time_out_buf
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids,
                         0:3] = 2.0*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        self.root_states[env_ids,
                         7:10] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        self.root_states[env_ids,
                         10:13] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        # self.root_states[env_ids,
        #                  0:3] = .0*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        # self.root_states[env_ids,
        #                  7:10] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        # self.root_states[env_ids,
        #                  10:13] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        self.root_states[env_ids, 3:7] = 0
        self.root_states[env_ids, 6] = 1.0

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        self.reset_buf[env_ids] = 1
        self.progress_buf[env_ids] = 0

        self.thrust_cmds_damp[env_ids] = 0
        self.thrust_rot_damp[env_ids] = 0

    def pre_physics_step(self, _actions):
        # resets
        if self.counter % 250 == 0:
            print("self.counter:", self.counter)
        self.counter += 1

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        actions = _actions.to(self.device)
        # tensor [n,4]
        obs_buf_cpu = self.obs_buf.cpu().numpy()
        root_pos_cpu = self.obs_buf[..., 0:3].cpu().numpy()
        root_quats_cpu = self.obs_buf[..., 3:7].cpu().numpy()
        lin_vel_cpu = self.obs_buf[..., 7:10].cpu().numpy()
        ang_vel_cpu = self.obs_buf[..., 10:13].cpu().numpy()

        # for vel_control
        self.actions = tensor_clamp(
            actions, self.action_lower_limits, self.action_upper_limits)
        actions_cpu = self.actions.cpu().numpy()

        root_quats_cpu = root_quats_cpu[:, [3, 0, 1, 2]]
        self.parallel_vel_control.set_status(root_pos_cpu,root_quats_cpu,lin_vel_cpu,ang_vel_cpu,0.01)
        self.cmd_thrusts = torch.tensor(self.parallel_vel_control.update(actions_cpu.astype(np.float64)))
        
        # for rate _control
        # self.actions = torch.clamp(actions, min=-1.0, max=1.0)
        # self.actions[...,:3] = self.actions[...,:3]*2.0 # 0 1 2 rad/s
        # self.actions[...,3] = (actions[...,3] + 1)/3 # 3 thrust
        # actions_cpu = self.actions.cpu().numpy()

        # root_quats_cpu = root_quats_cpu[:, [3, 0, 1, 2]]
        # self.parallel_rate_control.set_q_world(root_quats_cpu.astype(np.float64))
        # self.cmd_thrusts = torch.tensor(self.parallel_rate_control.update(actions_cpu.astype(np.float64),ang_vel_cpu.astype(np.float64),0.01)) 
        # end

        # debugging
        # 使用 torch.isnan() 检查张量中的元素是否为 NaN
        nan_mask = torch.isnan(self.cmd_thrusts)
        # 如果有 NaN，则进行暂停
        if nan_mask.any():
            print("Sleeping for 10 second...")
            time.sleep(10)
        
        thrusts=((self.cmd_thrusts**2)*5.0).to('cuda') # [n,4]

        force_x = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        force_y = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        force_xy = torch.cat((force_x, force_y), 1).reshape(-1, 4, 2)
        thrusts = thrusts.reshape(-1, 4, 1)
        thrusts = torch.cat((force_xy, thrusts), 2)

        self.thrusts = thrusts

        # # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0
        # # spin spinning rotors
        prop_rot = ((self.cmd_thrusts**2)*0.2).to('cuda')

        # prop_rot = self.thrust_cmds_damp * self.prop_max_rot
        self.torques[:, 1, 2] = -prop_rot[:, 0]
        self.torques[:, 2, 2] = -prop_rot[:, 1]
        self.torques[:, 3, 2] = prop_rot[:, 2]
        self.torques[:, 4, 2] = prop_rot[:, 3]

        self.forces[:, 1:5] = self.thrusts

        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(
            self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)
        # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(
        #     self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.GLOBAL_SPACE)
        # apply propeller rotation
        # self.gym.set_joint_target_velocity(self.sim, )

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def compute_observations(self):
        self.obs_buf[..., :3] = self.root_positions
        self.obs_buf[..., 3:7] = self.root_quats
        self.obs_buf[..., 7:10] = self.root_linvels
        self.obs_buf[..., 10:13] = self.root_angvels
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_quadcopter_reward(
            self.root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )


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
def compute_quadcopter_reward(root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # distance to target
    target_dist = torch.sqrt(root_positions[..., 0] * root_positions[..., 0] +
                             root_positions[..., 1] * root_positions[..., 1] +
                             (root_positions[..., 2]) * (root_positions[..., 2]))
    pos_reward = 2.0 / (1.0 + target_dist * target_dist)

    dist_reward = (20.0 - target_dist) / 40.0

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward) + dist_reward

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    # die = torch.where(target_dist > 10.0, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)
    reset = torch.where(torch.norm(root_positions, dim=1) > 10.0, ones, reset)

    return reward, reset
