import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET

from airgym import AIRGYM_ROOT_DIR, AIRGYM_ROOT_DIR

from isaacgym import gymutil, gymtorch, gymapi
from airgym.utils.torch_utils import *
from airgym.envs.base.X152bPx4_with_cam import X152bPx4WithCam
import airgym.utils.rotations as rot_utils
from airgym.envs.task.X152b_planning_config import X152bPlanningConfig
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

def compute_yaw_diff(a: torch.Tensor, b: torch.Tensor):
    """Compute the difference between two sets of Euler angles. a & b in [-pi, pi]"""
    diff = b - a
    diff = torch.where(diff < -torch.pi, diff + 2*torch.pi, diff)
    diff = torch.where(diff > torch.pi, diff - 2*torch.pi, diff)
    return diff

class X152bPlanning(X152bPx4WithCam):

    def __init__(self, cfg: X152bPlanningConfig, sim_params, physics_engine, sim_device, headless):
        self.cam_resolution = cfg.env.cam_resolution # set camera resolution
        self.cam_channel = cfg.env.cam_channel # set camera channel
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cam_resolution = cfg.env.cam_resolution # recover camera resolution
        self.cam_channel = cfg.env.cam_channel # recover camera channel

        if self.cfg.env.enable_onboard_cameras:
            print("Onboard cameras enabled...")
            print("Checking camera resolution =========== ", self.cam_resolution)
            self.full_camera_array = torch.zeros((self.num_envs, self.cam_channel, self.cam_resolution[0], self.cam_resolution[1]), device=self.device) # 1 for depth

        self.full_camera_array = torch.zeros((self.num_envs, self.cam_channel, self.cam_resolution[0], self.cam_resolution[1]), device=self.device) # 1 for depth

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

    def _create_envs(self):
        print("\n\n\n\n\n CREATING ENVIRONMENT \n\n\n\n\n\n")
        asset_path = self.cfg.asset_config.X152b.file.format(
            AIRGYM_ROOT_DIR=AIRGYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = asset_class_to_AssetOptions(self.cfg.asset_config.X152b)

        X152b = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(X152b)

        start_pose = gymapi.Transform()
        # create env instance
        pos = torch.tensor([0, 0, 0], device=self.device)
        start_pose.p = gymapi.Vec3(*pos)
        self.env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-self.env_spacing, - self.env_spacing, -self.env_spacing)
        env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)
        self.actor_handles = []
        self.env_asset_handles = []
        self.envs = []
        self.camera_handles = []
        self.camera_tensors = []

        # Set Camera Properties
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = self.cam_resolution[0]
        camera_props.height = self.cam_resolution[1]
        camera_props.far_plane = 5.0
        camera_props.horizontal_fov = 87.0
        camera_props.use_collision_geometry = True
        
        # local camera transform
        local_transform = gymapi.Transform()
        # position of the camera relative to the body
        local_transform.p = gymapi.Vec3(0.15, 0.00, 0.1)
        # orientation of the camera relative to the body
        # local_transform.r = gymapi.Quat(0.0, 0.269, 0.0, 0.963)
         
        local_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.segmentation_counter = 0

        for i in range(self.num_envs):
            # create environment
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # insert robot asset
            actor_handle = self.gym.create_actor(env_handle, X152b, start_pose, "robot", i, self.cfg.asset_config.X152b.collision_mask, 0)
            # append to lists
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            if self.enable_onboard_cameras:
                cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                self.gym.attach_camera_to_body(cam_handle, env_handle, actor_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
                self.camera_handles.append(cam_handle)
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor) # (height, width)
                # print("camera tensor shape: ", torch_cam_tensor.shape)
                self.camera_tensors.append(torch_cam_tensor)

                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # out = cv2.VideoWriter('depth_video.mp4', fourcc, 30, (270, 480))
                # self.outs.append(out)

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

        self.robot_bodies = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
        self.robot_mass = 0
        for body in self.robot_bodies:
            self.robot_mass += body.mass
        print("Total robot mass: ", self.robot_mass)
        
        print("\n\n\n\n\n ENVIRONMENT CREATED \n\n\n\n\n\n")

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        self.env_asset_manager.calculate_randomize_pose()
        self.env_asset_manager.calculate_specify_pose()

        # randomize asset root states
        self.env_asset_root_states[env_ids, :, 0:1] = 2. * torch_rand_float(-1.0, 1.0, (num_resets, self.num_assets, 1), self.device) + torch.tensor([4.], device=self.device)
        self.env_asset_root_states[env_ids, :, 1:2] = 2. * torch_rand_float(-1.0, 1.0, (num_resets, self.num_assets, 1), self.device) + torch.tensor([0.], device=self.device)
        self.env_asset_root_states[env_ids, :, 2:3] = 0
        assets_root_angle = torch.concatenate([0 * torch_rand_float(-torch.pi, torch.pi, (num_resets, self.num_assets, 2), self.device),
                                       torch_rand_float(-torch.pi, torch.pi, (num_resets, self.num_assets, 1), self.device)], dim=-1)
        assets_matrix = T.euler_angles_to_matrix(assets_root_angle, 'XYZ')
        assets_root_quats = T.matrix_to_quaternion(assets_matrix)
        self.env_asset_root_states[env_ids, :, 3:7] = assets_root_quats[:, :, [1, 2, 3, 0]]

        # randomize root states
        self.root_states[env_ids, 0:2] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device) + torch.tensor([0., 0.], device=self.device)
        self.root_states[env_ids, 2:3] = 0.2*torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1.

        # randomize root orientation
        root_angle = torch.concatenate([0.01*torch_rand_float(-torch.pi, torch.pi, (num_resets, 2), self.device), # .01
                                       0.05*torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device)], dim=-1) # 0.05

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

        # obs = self.obs_buf
        return obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_observations(self):
        self.root_matrix = T.quaternion_to_matrix(self.root_quats[:, [3, 0, 1, 2]]).reshape(self.num_envs, 9)
        self.obs_buf[..., 0:9] = self.root_matrix
        self.obs_buf[..., 9:12] = self.root_positions
        self.obs_buf[..., 12:15] = self.root_linvels
        self.obs_buf[..., 15:18] = self.root_angvels
        self.add_noise()
        self.obs_buf[..., 0:18] -= self.target_states
        
        self.obs_buf[..., 18:22] = self.actions
        # self.obs_buf[..., 22:34] = torch.rand((self.num_envs, 12), device=self.device)

        # self.full_camera_array = torch.rand((self.num_envs, self.cam_channel, self.cam_resolution[0], self.cam_resolution[1]), device=self.device)
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] ,self.item_reward_info = self.compute_quadcopter_reward()
        # update prev
        self.pre_actions = self.actions.clone()
        self.pre_root_positions = self.root_positions.clone()
        self.pre_root_angvels = self.root_angvels.clone()

    def compute_quadcopter_reward(self):
        # effort reward
        thrust_cmds = torch.clamp(self.cmd_thrusts, min=0.0, max=1.0).to('cuda')
        effort_reward = .1 * (1 - thrust_cmds).sum(-1)/4

        # continous actions
        action_diff = self.actions - self.pre_actions
        if self.ctl_mode == "pos" or self.ctl_mode == 'vel':
            continous_action_reward =  .2 * torch.exp(-torch.norm(action_diff[..., :], dim=-1))
        else:
            continous_action_reward = .1 * torch.exp(-torch.norm(action_diff[..., :-1], dim=-1)) + .5 / (1.0 + torch.square(2 * action_diff[..., -1]))
            thrust = self.actions[..., -1] # this thrust is the force on vertical axis
            thrust_reward = .1 * (1-torch.abs(0.1533 - thrust))
        
        # guidance reward
        x_linvel = self.root_linvels[:, 0]
        x_linvel_reward = x_linvel * torch.exp(1-x_linvel)
        y_diff_reward = 1 / (1.0 + torch.square(2 * self.root_positions[:, 1]))

        # height reward
        height_diff = 1. - self.root_positions[:, -1]
        height_reward = 1. / (1.0 + torch.square(1.8 * height_diff))

        # heading reward
        root_matrix = T.quaternion_to_matrix(self.root_quats[:, [3, 0, 1, 2]])
        root_euler = T.matrix_to_euler_angles(root_matrix, convention='XYZ')
        vel_dir = torch.arctan2(self.root_linvels[..., 1], self.root_linvels[..., 0])
        yaw_diff = compute_yaw_diff(vel_dir, root_euler[..., 2]) / torch.pi
        yaw_reward = 1 / (1.0 + torch.square(2 * yaw_diff))

        # uprightness
        ups = quat_axis(self.root_quats, 2)
        ups_reward = torch.square((ups[..., 2] + 1) / 2)

        # collision
        alive_reward = torch.where(self.collisions > 0, -500., 0.5)

        reward = (
            continous_action_reward
            + effort_reward
            + thrust_reward
            + height_reward
            + continous_action_reward * (x_linvel_reward + yaw_reward + ups_reward + y_diff_reward)
            + alive_reward
        )

        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        # resets due to episode length
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, ones, die)
        # print(self.progress_buf)

        # resets due to huge x vel
        reset = torch.where(self.root_linvels[:, 0] > 1.2, ones, reset)

        # resets due to negative x vel
        reset = torch.where(self.root_linvels[:, 0] < 0, ones, reset)

        # resets due to huge y diff
        reset = torch.where(self.root_positions[:, 1] > 1, ones, reset)
        reset = torch.where(self.root_positions[:, 1] < -1, ones, reset)
        # print(self.root_positions[:, 1])

        # resets due to z diff
        reset = torch.where(self.root_positions[:, 2] < 0.5, ones, reset)
        reset = torch.where(self.root_positions[:, 2] > 1.5, ones, reset)

        # resets due to a negative w in quaternions
        if self.ctl_mode == "atti":
            reset = torch.where(self.actions[..., 0] < 0, ones, reset)
        
        item_reward_info = {}
        item_reward_info["x_linvel_reward"] = x_linvel_reward
        item_reward_info["y_diff_reward"] = y_diff_reward
        item_reward_info["yaw_reward"] = yaw_reward
        item_reward_info["continous_action_reward"] = continous_action_reward
        item_reward_info["thrust_reward"] = thrust_reward
        item_reward_info["effort_reward"] = effort_reward
        item_reward_info["ups_reward"] = ups_reward
        item_reward_info["height_reward"] = height_reward
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