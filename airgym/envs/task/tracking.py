import torch

from isaacgym import gymtorch, gymapi
from airgym.utils.torch_utils import *
from airgym.envs.task.tracking_config import TrackingCfg
from airgym.envs.base.hovering import Hovering
from airgym.assets.asset_manager import AssetManager

from rlPx4Controller.pyParallelControl import ParallelRateControl,ParallelVelControl,ParallelAttiControl,ParallelPosControl

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

class Tracking(Hovering):

    def __init__(self, cfg: TrackingCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        assert cfg.env.ctl_mode is not None, "Please specify one control mode!"
        print("ctl mode =========== ", cfg.env.ctl_mode)
        self.ctl_mode = cfg.env.ctl_mode
        self.cfg.env.num_actions = 5 if cfg.env.ctl_mode == "atti" else 4
        self.episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.debug_viz = False
        num_actors = 1

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless

        self.asset_manager = AssetManager(self.cfg, sim_device)

        super(Hovering, self).__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        self.contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        num_actors = self.asset_manager.get_env_actor_count() # Number of actors including robots and env assets in the environment
        num_env_assets = self.asset_manager.get_env_asset_count() # Number of env assets
        robot_num_bodies = self.asset_manager.get_robot_num_bodies() # Number of robots bodies in environment
        env_asset_link_count = self.asset_manager.get_env_asset_link_count() # Number of env assets links in the environment
        env_boundary_count = self.asset_manager.get_env_boundary_count() # Number of env boundaries in the environment
        self.num_assets = num_env_assets - env_boundary_count # # Number of env assets that can be randomly placed
        bodies_per_env = env_asset_link_count + robot_num_bodies

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

        # set target states
        self.target_states = torch.tensor(self.cfg.env.target_state, device=self.device).repeat(self.num_envs, 1)

        # actions
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.pre_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

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

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # set drone root state
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # randomize root states
        self.root_states[env_ids, 0:2] = .1*torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device)
        self.root_states[env_ids, 2:3] = .1*torch_rand_float(-1., 1., (num_resets, 1), self.device) + 1.

        # randomize root orientation
        root_angle = torch.concatenate([0.1*torch_rand_float(-torch.pi, torch.pi, (num_resets, 2), self.device), # .1
                                       0.2*torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device)], dim=-1) # 0.2

        matrix = T.euler_angles_to_matrix(root_angle, 'XYZ')
        root_quats = T.matrix_to_quaternion(matrix) # w,x,y,z
        self.root_states[env_ids, 3:7] = root_quats[:, [1, 2, 3, 0]] #x,y,z,w

        # randomize root linear and angular velocities
        self.root_states[env_ids, 7:10] = 0.5*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device) # 0.5
        self.root_states[env_ids, 10:13] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device) # 0.2

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        self.reset_buf[env_ids] = 1
        self.progress_buf[env_ids] = 0

        self.thrust_cmds_damp[env_ids] = 0
        self.thrust_rot_damp[env_ids] = 0

        self.int_pos_error[env_ids] = 0
        self.int_yaw_error[env_ids] = 0

        self.pre_actions[env_ids] = 0
        self.pre_root_positions[env_ids] = 0

    def compute_traj_lemniscate(self, n_steps=10, step_size=5, scale=0.25):
        step = self.progress_buf.unsqueeze(1).expand(-1, n_steps) + torch.arange(n_steps, device=self.device).repeat(self.num_envs, 1) * step_size
        t = step * self.dt * scale
        ref_x = 3 * torch.sin(t) / (1 + torch.cos(t) ** 2)
        ref_y = 3 * torch.sin(t) * torch.cos(t) / (1 + torch.cos(t) ** 2)
        ref_z = torch.ones_like(ref_x)
        return torch.stack((ref_x, ref_y, ref_z), dim=-1)

    def compute_observations(self):
        self.root_matrix = T.quaternion_to_matrix(self.root_quats[:, [3, 0, 1, 2]]).reshape(self.num_envs, 9)
        self.obs_buf[..., 0:9] = self.root_matrix
        self.obs_buf[..., 9:12] = self.root_positions
        self.obs_buf[..., 12:15] = self.root_linvels
        self.obs_buf[..., 15:18] = self.root_angvels

        self.ref_positions = self.compute_traj_lemniscate()
        self.related_future_pos = (self.ref_positions - self.root_positions.clone().unsqueeze(1)).reshape(self.num_envs, -1)
        self.obs_buf[..., 18:48] = self.related_future_pos

        self.add_noise()
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] ,self.item_reward_info = self.compute_quadcopter_reward()
        
        # update prev
        self.pre_actions = self.actions.clone()
        self.pre_root_positions = self.root_positions.clone()

    def compute_quadcopter_reward(self):
        # effort reward
        thrust_cmds = torch.clamp(self.cmd_thrusts, min=0.0, max=1.0).to('cuda')
        effort_reward = .1 * (1 - thrust_cmds).sum(-1)/4

        # continous actions
        action_diff = self.actions - self.pre_actions
        if self.ctl_mode == "pos" or self.ctl_mode == 'vel' or self.ctl_mode == 'prop':
            continous_action_reward =  .2 * torch.exp(-torch.norm(action_diff[..., :], dim=-1))
        else:
            continous_action_reward = .1 * torch.exp(-torch.norm(action_diff[..., :-1], dim=-1)) + .5 / (1.0 + torch.square(2 * action_diff[..., -1]))
            thrust = self.actions[..., -1] # this thrust is the force on vertical axis
            thrust_reward = .1 * (1-torch.abs(0.1533 - thrust))
        
        # dist reward
        dist_diff = self.ref_positions[:, 0]-self.root_positions
        dist_norm = torch.norm(dist_diff, dim=-1)
        dist_reward = 1. / (1.0 + torch.square(1.8 * dist_norm))

        # heading reward
        target_matrix = self.target_states[..., 0:9].reshape(self.num_envs, 3,3)
        target_euler = T.matrix_to_euler_angles(target_matrix, 'XYZ')
        root_matrix = T.quaternion_to_matrix(self.root_quats[:, [3, 0, 1, 2]])
        root_euler = T.matrix_to_euler_angles(root_matrix, convention='XYZ')
        yaw_diff = compute_yaw_diff(target_euler[..., 2], root_euler[..., 2]) / torch.pi
        yaw_reward = 1 / (1.0 + torch.square(4 * yaw_diff))

        spinnage = torch.square(self.root_angvels[:, -1])
        spin_reward = 1 / (1.0 + torch.square(2 * spinnage))

        # uprightness
        ups = quat_axis(self.root_quats, 2)
        ups_reward = torch.square((ups[..., 2] + 1) / 2)

        if self.ctl_mode == "pos" or self.ctl_mode == 'vel' or self.ctl_mode == 'prop':
            reward = (
                continous_action_reward
                + effort_reward
                + dist_reward
                + dist_reward * (spin_reward + yaw_reward + ups_reward)
            )
        else:
            reward = (
                continous_action_reward
                + effort_reward
                + thrust_reward
                + dist_reward
                + dist_reward * (spin_reward + yaw_reward + ups_reward)
            )

        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        # resets due to episode length
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, ones, die)
        reset = torch.where(dist_norm > 1.0, ones, reset)

        # resets due to a negative w in quaternions
        if self.ctl_mode == "atti":
            reset = torch.where(self.actions[..., 0] < 0, ones, reset)
        
        item_reward_info = {}
        item_reward_info["dist_norm"] = dist_norm
        item_reward_info["dist_reward"] = dist_reward
        item_reward_info["yaw_reward"] = yaw_reward
        item_reward_info["spin_reward"] = spin_reward
        item_reward_info["continous_action_reward"] = continous_action_reward
        item_reward_info["thrust_reward"] = thrust_reward if self.ctl_mode == "atti" or self.ctl_mode == 'rate' else 0
        item_reward_info["effort_reward"] = effort_reward
        item_reward_info["ups_reward"] = ups_reward
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