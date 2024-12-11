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

from rlPx4Controller.pyParallelControl import ParallelRateControl,ParallelVelControl,ParallelAttiControl,ParallelPosControl

import matplotlib.pyplot as plt
from airgym.utils.helpers import asset_class_to_AssetOptions
from airgym.utils.rotations import quats_to_euler_angles
import time

import pytorch3d.transforms as T

import rospy
from std_msgs.msg import Float64MultiArray

# #---- parameters for outter loop (vel/pos) training ----##
# C1 = 1                # for continous actions
# C2 = 0               # not used
# TH = 0               # not used
# P1 = 2             # for horizental position
# P2 = 3               # for horizental position accuracy
# P3 = 1               # for vertical position
# P4 = 6               # for vertical position accuracy
# P5 = 0.2             # for position error
# V1 = 1             # for velocity
# V2 = 6               # for velocity accuracy
# V3 = 0.5             # for velocity direction
# Y1 = 1               # for yaw
# Y2 = 3               # for yaw accuracy
# Y3 = 0.1             # for yaw error
# A1 = 0.18            # for rate
# A2 = 6               # for rate accuracy
# E = 0.4              # for energy consumption

##---- parameters for inner loop (atti/rate/prop) training ----##
C1 = 1.2                # for continous actions
C2 = 1.2                # for continous thrust
TH = 1                # for thrust to overcome gravity
P1 = 1              # for horizental position
P2 = 6               # for horizental position accuracy
P3 = 1               # for vertical position
P4 = 6               # for vertical position accuracy
P5 = 1               # for position error
V1 = 1               # for velocity
V2 = 4               # for velocity accuracy
V3 = 0.5             # for velocity direction
Y1 = 1.5               # for yaw
Y2 = torch.pi        # for yaw accuracy
Y3 = 0.3               # for yaw error
A1 = 1               # for rate
A2 = 6               # for rate accuracy
E = 0.4              # for energy consumption

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

class X152bPx4(BaseTask):

    def __init__(self, cfg: X152bPx4Cfg, sim_params, physics_engine, sim_device, headless):
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

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        bodies_per_env = self.robot_num_bodies

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

        # controller
        self.cmd_thrusts = torch.zeros((self.num_envs, 4))
        # choice 1 from rate ctrl and vel ctrl
        if(cfg.env.ctl_mode == "pos"):
            self.action_upper_limits = torch.tensor(
            [3, 3, 3, 6.0], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor(
            [-3, -3, -3, -6.0], device=self.device, dtype=torch.float32)
            self.parallel_pos_control = ParallelPosControl(self.num_envs)
        elif(cfg.env.ctl_mode == "vel"):
            self.action_upper_limits = torch.tensor(
                [6, 6, 6, 6], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor(
                [-6, -6, -6, -6], device=self.device, dtype=torch.float32)
            self.parallel_vel_control = ParallelVelControl(self.num_envs)
        elif(cfg.env.ctl_mode == "atti"): # w, x, y, z, thrust
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

        # set target states
        self.target_states = torch.tensor(self.cfg.env.target_state, device=self.device).repeat(self.num_envs, 1)

        # actions
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.pre_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        # reward integration buffers
        self.int_pos_error = torch.zeros((self.num_envs, 10), device=self.device)
        self.int_yaw_error = torch.zeros((self.num_envs, 10), device=self.device)

        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env
            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # test ros actions
        # rospy.init_node('ctl_onboard', anonymous=True)
        # self.pub = rospy.Publisher('/action', Float64MultiArray, queue_size=10)
        # self.sub = rospy.Subscriber('/target_state', Float64MultiArray, self.callback)
        
        if self.cfg.use_tcn:
            self.tcn_seqs_len = self.cfg.tcn_seqs_len
            self.obs_seqs_buf = torch.zeros(
                (self.num_envs, self.tcn_seqs_len, self.cfg.env.num_observations), device=self.device, dtype=torch.float32)

    def callback(self, data):
        self.target_state = torch.tensor(data.data, device=self.device)
        self.target_states = self.target_state.repeat(self.num_envs, 1)

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
        asset_path = self.cfg.asset_config.X152b.file.format(
            AIRGYM_ROOT_DIR=AIRGYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = asset_class_to_AssetOptions(self.cfg.asset_config.X152b)

        X152b = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)

        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(X152b)

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
                env_handle, X152b, start_pose, self.cfg.asset_config.X152b.name, i, self.cfg.asset_config.X152b.collision_mask, 0)
            
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
        self.extras["item_reward_info"] = self.item_reward_info
        
        if self.cfg.use_tcn: # use TCN
            self.obs_seqs_buf = torch.cat(
                (self.obs_seqs_buf[:, 1:], self.obs_buf.unsqueeze(1)), dim=1)
            return self.obs_seqs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # randomize root states
        self.root_states[env_ids, 0:2] = 2.0*torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device) # 2.0
        self.root_states[env_ids, 2] = 2.0*torch_one_rand_float(-1., 1., (num_resets, 1), self.device).squeeze(-1) # 2
        # self.root_states[env_ids, 0] = 0 # debug
        # self.root_states[env_ids, 1] = 0 # debug
        # self.root_states[env_ids, 2] = 0 # debug

        # randomize root orientation
        root_angle = torch.concatenate([0.1*torch_rand_float(-torch.pi, torch.pi, (num_resets, 2), self.device), # .1
                                       0.2*torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device)], dim=-1) # 0.2
        # root_angle = torch.concatenate([0.*torch.ones((num_resets, 1), device=self.device), # debug
        #                                 0.*torch.ones((num_resets, 1), device=self.device), # debug
        #                                 0.8*torch.pi*torch.ones((num_resets, 1), device=self.device)], dim=-1) # debug
        matrix = T.euler_angles_to_matrix(root_angle, 'XYZ')
        root_quats = T.matrix_to_quaternion(matrix) # w,x,y,z
        self.root_states[env_ids, 3:7] = root_quats[:, [1, 2, 3, 0]] #x,y,z,w

        # randomize root linear and angular velocities
        self.root_states[env_ids, 7:10] = 0.5*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device) # 0.5
        self.root_states[env_ids, 10:13] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device) # 0.2
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

        if self.cfg.use_tcn:
            self.obs_seqs_buf[env_ids] = 0

    def pre_physics_step(self, _actions):
        # resets
        if self.counter % 250 == 0:
            print("self.counter:", self.counter)
        self.counter += 1

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        actions = _actions.to(self.device)

        # debugging
        # if self.cfg.env.ctl_mode == "atti":
        #     print("actions:", actions[0])
            # assert torch.all(actions[..., 0] >= 0), "w in q must be positive!"
        
        self.actions = tensor_clamp(
            actions, self.action_lower_limits, self.action_upper_limits)
        
        actions_cpu = self.actions.cpu().numpy()
        
        #--------------- input state for pid controller. tensor [n,4] --------#
        obs_buf_cpu = self.root_states.cpu().numpy()
        # pos
        root_pos_cpu = self.root_states[..., 0:3].cpu().numpy()
        # quat. if w is negative, then set it to positive. x,y,z,w
        self.root_states[..., 3:7] = torch.where(self.root_states[..., 6:7] < 0, 
                                                 -self.root_states[..., 3:7], 
                                                 self.root_states[..., 3:7])
        root_quats_cpu = self.root_states[..., 3:7].cpu().numpy() # x,y,z,w
        # lin vel
        lin_vel_cpu = self.root_states[..., 7:10].cpu().numpy()
        # ang vel
        ang_vel_cpu = self.root_states[..., 10:13].cpu().numpy()

        # print(actions)
        control_mode_ = self.ctl_mode
        if(control_mode_ == "pos"):
            root_quats_cpu = root_quats_cpu[:, [3, 0, 1, 2]]
            self.parallel_pos_control.set_status(root_pos_cpu,root_quats_cpu,lin_vel_cpu,ang_vel_cpu,0.01)
            self.cmd_thrusts = torch.tensor(self.parallel_pos_control.update(actions_cpu.astype(np.float64)))
        elif(control_mode_ == "vel"):
            root_quats_cpu = root_quats_cpu[:, [3, 0, 1, 2]]
            self.parallel_vel_control.set_status(root_pos_cpu,root_quats_cpu,lin_vel_cpu,ang_vel_cpu,0.01)
            self.cmd_thrusts = torch.tensor(self.parallel_vel_control.update(actions_cpu.astype(np.float64)))
        elif(control_mode_ == "atti"):
            root_quats_cpu = root_quats_cpu[:, [3, 0, 1, 2]] # w, x, y, z
            self.parallel_atti_control.set_status(root_pos_cpu,root_quats_cpu,lin_vel_cpu,ang_vel_cpu,0.01)
            self.cmd_thrusts = torch.tensor(self.parallel_atti_control.update(actions_cpu.astype(np.float64))) 
        elif(control_mode_ == "rate"):
            root_quats_cpu = root_quats_cpu[:, [3, 0, 1, 2]]
            self.parallel_rate_control.set_q_world(root_quats_cpu.astype(np.float64))
            self.cmd_thrusts = torch.tensor(self.parallel_rate_control.update(actions_cpu.astype(np.float64),ang_vel_cpu.astype(np.float64),0.01)) 
        elif(control_mode_ == "prop"):
            self.cmd_thrusts =  self.actions
        else:
            print("Mode error")
        # end

        # debugging
        # 使用 torch.isnan() 检查张量中的元素是否为 NaN
        # nan_mask = torch.isnan(self.cmd_thrusts)
        # # 如果有 NaN，则进行暂停
        # if nan_mask.any():
        #     print("Sleeping for 10 second...")
        #     time.sleep(10)
        
        thrusts=((self.cmd_thrusts**2)*9.57).to('cuda')

        force_x = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        force_y = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        force_xy = torch.cat((force_x, force_y), 1).reshape(-1, 4, 2)
        thrusts = thrusts.reshape(-1, 4, 1)
        thrusts = torch.cat((force_xy, thrusts), 2)

        self.thrusts = thrusts

        # # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0
        # # spin spinning rotors
        prop_rot = ((self.cmd_thrusts)*0.2).to('cuda')

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
        self.root_matrix = T.quaternion_to_matrix(self.root_quats[:, [3, 0, 1, 2]]).reshape(self.num_envs, 9)
        # print(self.root_matrix)
        self.obs_buf[..., 0:9] = self.root_matrix
        self.obs_buf[..., 9:12] = self.root_positions
        self.obs_buf[..., 12:15] = self.root_linvels
        self.obs_buf[..., 15:18] = self.root_angvels

        self.add_noise()

        if not self.cfg.controller_test:
            self.obs_buf -= self.target_states

        return self.obs_buf

    def add_noise(self):
        matrix_noise = 1e-3 *torch_normal_float((self.num_envs, 9), self.device)
        pos_noise = 5e-3 *torch_normal_float((self.num_envs, 3), self.device)
        linvels_noise = 2e-2 *torch_normal_float((self.num_envs, 3), self.device)
        angvels_noise = 4e-1 *torch_normal_float((self.num_envs, 3), self.device)

        self.obs_buf[..., 0:9] += matrix_noise
        self.obs_buf[..., 9:12] += pos_noise
        self.obs_buf[..., 12:15] += linvels_noise
        self.obs_buf[..., 15:18] += angvels_noise

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] ,self.item_reward_info= self.compute_quadcopter_reward(
            self.ctl_mode,
            self.actions,
            self.pre_actions,
            self.cmd_thrusts,
            self.root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.max_episode_length, 
            self.target_states
        )
        action_data = Float64MultiArray()
        action_data.data = [self.actions[0,0].item(),self.actions[0,1].item(),self.actions[0,2].item(),self.actions[0,3].item()]
        
        # ros target pub
        # self.pub.publish(action_data)
        
        # update prev 
        self.pre_actions = self.actions.clone()
    
    def compute_quadcopter_reward(self, ctrl_mode, actions, pre_actions, cmd_thrusts, root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length, target_states):
        # type: (Str, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor,Dict[str, Tensor]]
        
        # continous action
        action_diff = actions - pre_actions
        if ctrl_mode == "pos" or ctrl_mode == 'vel':
            continous_action_reward =  C1 * (1 - torch.sqrt(action_diff.pow(2).sum(-1))/5)
        else:
            continous_action_reward = C1 * (1- torch.sqrt(action_diff[..., :-1].pow(2).sum(-1))/5) + C2 * (1-torch.sqrt(action_diff[..., -1].pow(2))/5)
            thrust = actions[..., -1] # this thrust is the force on vertical axis
            thrust_reward = TH * (1-torch.abs(0.1533 - thrust))

        # distance
        target_positions = target_states[..., 9:12]
        relative_positions = target_positions - root_positions
        pos_diff_h = torch.sqrt(relative_positions[..., 0] * relative_positions[..., 0] +
                                relative_positions[..., 1] * relative_positions[..., 1])
        pos_diff_v = torch.sqrt(relative_positions[..., 2] * relative_positions[..., 2])
        
        self.int_pos_error[..., 1:] = self.int_pos_error[..., :-1]
        self.int_pos_error[..., 0] = pos_diff_h + pos_diff_v

        pos_reward = P1 * (1.0 - 1/P2*pos_diff_h) + P3 * (1.0 - 1/P4*pos_diff_v) 
        pos_error_reward = P5 * (1-self.int_pos_error.sum(-1) / 35)
        _pos_reward = pos_reward + pos_error_reward

        # velocity
        target_linvels = target_states[..., 12:15]
        relative_linvels = root_linvels - target_linvels
        vel_diff = torch.norm(relative_linvels, dim=1)
        vel_reward = V1 * (1-(1/V2)*vel_diff)

        # velocity direction
        tar_direction = relative_positions / torch.norm(relative_positions, dim=1, keepdim=True)
        vel_direction = root_linvels / torch.norm(root_linvels, dim=1, keepdim=True)
        dot_product = (tar_direction * vel_direction).sum(dim=1)
        angle_difference = torch.acos(dot_product.clamp(-1.0, 1.0)).abs()
        vel_direction_error_reward = V3 * (1 - angle_difference / torch.pi)
        _vel_reward = vel_reward + vel_direction_error_reward

        # yaw
        target_matrix = target_states[..., 0:9].reshape(self.num_envs, 3,3)
        target_euler = T.matrix_to_euler_angles(target_matrix, 'XYZ')

        root_matrix = T.quaternion_to_matrix(root_quats[:, [3, 0, 1, 2]])
        root_euler = T.matrix_to_euler_angles(root_matrix, convention='XYZ')

        yaw_diff = torch.abs(compute_yaw_diff(target_euler[..., 2], root_euler[..., 2]))
        yaw_reward = Y1 * (1. - (1./Y2)*yaw_diff)

        self.int_yaw_error[..., 1:] = self.int_yaw_error[..., :-1]
        self.int_yaw_error[..., 0] = yaw_diff
        yaw_error_reward = Y3 * (1 - self.int_yaw_error.sum(-1)/(torch.pi*10))
        _yaw_reward = yaw_reward + yaw_error_reward

        # angular velocity
        target_angvels = target_states[..., 15:18]
        relative_angvels = root_angvels - target_angvels
        angvel_diff = torch.norm(relative_angvels, dim=1)
        angvel_reward = A1 * (1.0 - (1/A2)*angvel_diff)

        # uprightness
        ups = quat_axis(root_quats, 2)

        # effort reward
        thrust_cmds = torch.clamp(cmd_thrusts, min=0.0, max=1.0).to('cuda')
        effort_reward = E * (1 - thrust_cmds).sum(-1)/4

        # combined reward
        if ctrl_mode == "vel" or ctrl_mode == "pos":
            reward = continous_action_reward + angvel_reward + _vel_reward + _pos_reward + effort_reward + _yaw_reward
        elif ctrl_mode == "atti" or ctrl_mode == "rate":
            reward = continous_action_reward + angvel_reward + _vel_reward + _pos_reward + effort_reward + _yaw_reward + thrust_reward
        else:
            reward = continous_action_reward + angvel_reward + _vel_reward + _pos_reward + effort_reward + _yaw_reward
        
        # reward = continous_action_reward + _pos_reward + effort_reward

        # resets due to misbehavior
        ones = torch.ones_like(reset_buf)
        die = torch.zeros_like(reset_buf)

        # resets due to episode length
        reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

        reset = torch.where(torch.norm(relative_positions, dim=1) > 4, ones, reset)
        
        reset = torch.where(torch.norm(relative_linvels, dim=1) > 6.0, ones, reset)
        
        reset = torch.where(relative_angvels[..., 2] > 17.5, ones, reset)
        reset = torch.where(relative_angvels[..., 2] < -17.5, ones, reset)
        
        reset = torch.where(relative_positions[..., 2] < -2, ones, reset)
        reset = torch.where(relative_positions[..., 2] > 2, ones, reset)

        reset = torch.where(ups[..., 2] < 0.0, ones, reset) # orient_z 小于0 = 飞行器朝下了

        # resets due to a negative w in quaternions
        if ctrl_mode == "atti":
            reset = torch.where(actions[..., 0] < 0, ones, reset)
        
        item_reward_info = {}
        item_reward_info["angvel_reward"] = angvel_reward
        item_reward_info["effort_reward"] = effort_reward
        item_reward_info["pos_reward"] = pos_reward
        item_reward_info["pos_error_reward"] = pos_error_reward
        item_reward_info["vel_reward"] = vel_reward
        item_reward_info["vel_direction_error_reward"] = vel_direction_error_reward
        item_reward_info["yaw_reward"] = yaw_reward
        item_reward_info["yaw_error_reward"] = yaw_error_reward
        item_reward_info["continous_action_reward"] = continous_action_reward

        if ctrl_mode == "atti"  or ctrl_mode == "rate":
            item_reward_info["thrust_reward"] = thrust_reward

        return reward, reset, item_reward_info


    # #---------------------- Original Reward Function ----------------------#
    # def compute_quadcopter_reward(self, ctrl_mode, actions, pre_actions, cmd_thrusts, root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length, target_states):
    #     # type: (Str, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor,Dict[str, Tensor]]
        
    #     # distance to target
    #     # target_positions = torch.tensor([0, 5, 1], dtype=torch.float32, device='cuda').unsqueeze(0).expand_as(root_positions)
    #     # relative_positions = root_positions - target_positions
    #     target_dist = torch.sqrt(root_positions[..., 0] * root_positions[..., 0] +
    #                             root_positions[..., 1] * root_positions[..., 1] +
    #                             (root_positions[..., 2]) * (root_positions[..., 2]))
    #     pos_reward = 2 * (1.0 - (1/6)*target_dist)

    #     target_vel = torch.norm(root_linvels, dim=1)
    #     target_ang = torch.norm(root_angvels, dim=1)

    #     vel_reward = 0.4 * (1-(1/6)*target_vel)
    #     ang_vel_reward =  0.2 * (1.0 - (1/6)*target_ang)

    #     # uprightness
    #     ups = quat_axis(root_quats, 2)

    #     # effort reward
    #     thrust_cmds = torch.clamp(cmd_thrusts, min=0.0, max=1.0).to('cuda')
    #     effort_reward = 0.4 * (1 - thrust_cmds).sum(-1)/4

    #     # combined reward
    #     reward = ang_vel_reward + vel_reward + pos_reward + effort_reward
    

    #     # resets due to misbehavior
    #     ones = torch.ones_like(reset_buf)
    #     die = torch.zeros_like(reset_buf)
    #     # die = torch.where(target_dist > 10.0, ones, die)

    #     # resets due to episode length
    #     reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)
    #     reset = torch.where(torch.norm(root_positions, dim=1) > 4.0, ones, reset)
    #     reset = torch.where(torch.norm(root_linvels, dim=1) > 6.0, ones, reset)

    #     reset = torch.where(root_positions[..., 2] < -2, ones, reset)
    #     reset = torch.where(root_positions[..., 2] > 2, ones, reset)

    #     reset = torch.where(ups[..., 2] < 0.0, ones, reset) # orient_z 小于0 = 飞行器朝下了
        
    #     item_reward_info = {}
    #     item_reward_info["ang_vel_reward"] = ang_vel_reward
    #     item_reward_info["effort_reward"] = effort_reward
    #     item_reward_info["pos_reward"] = pos_reward
    #     item_reward_info["vel_reward"] = vel_reward

    #     return reward, reset, item_reward_info


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