import torch

from isaacgym import gymapi, gymtorch
from airgym.utils.torch_utils import *
from airgym.envs.base.customized import Customized
from airgym.envs.task.maplanning_config import MAPlanningCfg
from airgym.assets.asset_manager import AssetManager

import pytorch3d.transforms as T
import torch.nn.functional as F
import cv2

from rlPx4Controller.pyParallelControl import ParallelRateControl,ParallelVelControl,ParallelAttiControl,ParallelPosControl


LENGTH =  8.0
WIDTH = 4.0
FLY_HEIGHT = 1.5 #1.0

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

class MAPlanning(Customized):

    def __init__(self, cfg: MAPlanningCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        assert cfg.env.ctl_mode is not None, "Please specify one control mode!"
        print("ctl mode =========== ", cfg.env.ctl_mode)
        self.ctl_mode = cfg.env.ctl_mode
        self.cfg.env.num_actions = 5 if cfg.env.ctl_mode == "atti" else 4
        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.debug_viz = False
        num_actors = 1

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless

        self.asset_manager = AssetManager(self.cfg, sim_device)
        super(Customized, self).__init__(cfg, sim_params, physics_engine, sim_device, headless)

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
        self.num_robots = num_robots = num_actors - num_env_assets
        cfg.env.agents = self.num_robots
        bodies_per_env = env_asset_link_count + robot_num_bodies

        self.obs_buf = torch.zeros(self.num_envs, self.num_robots, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, self.num_robots, device=self.device, dtype=torch.float)
        
        self.vec_root_tensor = gymtorch.wrap_tensor(
            self.root_tensor).view(self.num_envs, num_actors, 13)
        
        self.root_states = self.vec_root_tensor[:, :num_robots, :]
        self.root_positions = self.root_states[..., 0:3]
        self.root_quats = self.root_states[..., 3:7] # x,y,z,w
        self.root_linvels = self.root_states[..., 7:10]
        self.root_angvels = self.root_states[..., 10:13]
        
        if self.vec_root_tensor.shape[1] > num_robots:
            self.env_asset_root_states = self.vec_root_tensor[:, num_robots:, :]
            if self.get_privileged_obs:
                self.privileged_obs_buf = self.env_asset_root_states

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.counter = 0

        # setup rlPx4Controller
        self.cmd_thrusts = torch.zeros((self.num_envs, num_robots, 4))
        # choice 1 from rate ctrl and vel ctrl
        if(cfg.env.ctl_mode == "pos"):
            self.action_upper_limits = torch.tensor(
            [3, 3, 3, 6.0], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor(
            [-3, -3, -3, -6.0], device=self.device, dtype=torch.float32)
            self.parallel_pos_control = []
        elif(cfg.env.ctl_mode == "vel"):
            self.action_upper_limits = torch.tensor(
                [6, 6, 6, 6], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor(
                [-6, -6, -6, -6], device=self.device, dtype=torch.float32)
            self.parallel_vel_control = []
        elif(cfg.env.ctl_mode == "atti"): # w, x, y, z, thrust
            self.action_upper_limits = torch.tensor(
            [1, 1, 1, 1, 1], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor(
            [-1, -1, -1, -1, 0.], device=self.device, dtype=torch.float32)
            self.parallel_atti_control = []
        elif(cfg.env.ctl_mode == "rate"):
            self.action_upper_limits = torch.tensor(
                [1, 1, 1, 1], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor(
                [-1, -1, -1, 0], device=self.device, dtype=torch.float32)
            self.parallel_rate_control = []
        elif(cfg.env.ctl_mode == "prop"):
            self.action_upper_limits = torch.tensor(
                [1, 1, 1, 1], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor(
                [0, 0, 0, 0], device=self.device, dtype=torch.float32)
        else:
            print("Mode Error!")
        
        for i in range(num_robots):
            if cfg.env.ctl_mode == "pos":
                controller = ParallelPosControl(self.num_envs)
                self.parallel_pos_control.append(controller)
            elif cfg.env.ctl_mode == "vel":
                controller = ParallelVelControl(self.num_envs)
                self.parallel_vel_control.append(controller)
            elif cfg.env.ctl_mode == "atti":
                controller = ParallelAttiControl(self.num_envs)
                self.parallel_atti_control.append(controller)
            elif cfg.env.ctl_mode == "rate":
                controller = ParallelRateControl(self.num_envs)
                self.parallel_rate_control.append(controller)
            else:
                print("Prop mode needs no controller.")

        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3),
                                  dtype=torch.float32, device=self.device, requires_grad=False)
        self.torques = torch.zeros((self.num_envs, bodies_per_env, 3),
                                   dtype=torch.float32, device=self.device, requires_grad=False)

        # control parameters
        self.thrusts = torch.zeros((self.num_envs, num_robots, 4, 3), dtype=torch.float32, device=self.device)

        # set target states
        self.target_states = torch.tensor(self.cfg.env.target_state, device=self.device).view(1, 1, -1).expand(self.num_envs, self.num_robots, -1)

        # collision
        body_contact_forces = gymtorch.wrap_tensor(self.contact_force_tensor).view(self.num_envs, bodies_per_env, 3)
        indices = torch.arange(0, self.num_robots * 5, step=5, device=self.device)
        self.contact_forces = body_contact_forces[:, indices, :]
        self.collisions = torch.zeros(self.num_envs, self.num_robots, device=self.device)

        # actions
        self.actions = torch.zeros((self.num_envs, num_robots, self.num_actions), device=self.device)
        self.pre_actions = torch.zeros((self.num_envs, num_robots, self.num_actions), device=self.device)

        # reset
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_robot = torch.zeros((self.num_envs, num_robots), device=self.device, dtype=torch.long)

        if self.enable_onboard_cameras:
            print("Onboard cameras enabled...")
            print("camera resolution =========== ", self.cam_resolution)
            self.full_camera_array = torch.zeros((self.num_envs, num_robots, self.cam_channel, self.cam_resolution[0], self.cam_resolution[1]), device=self.device) # 1 for depth
            
        # get states of goal
        self.goal_states = self.env_asset_root_states[:, 0:1, :]
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

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs * self.num_robots, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
    
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
        self.goal_states[env_ids, :, 0:1] = torch.tensor([LENGTH+0.5], device=self.device)
        self.goal_states[env_ids, :, 1:2] = 1.5 * torch_rand_float(-1.0, 1.0, (num_resets, 1, 1), self.device) + torch.tensor([0.], device=self.device)
        self.goal_states[env_ids, :, 2:3] = .0 * torch_rand_float(-1., 1., (num_resets, 1, 1), self.device) + FLY_HEIGHT

        # randomize root states
        self.root_states[env_ids, :, 0:1] = .0 *torch_rand_float(-1., 1., (num_resets, self.num_robots, 1), self.device) + torch.tensor([-LENGTH-0.5], device=self.device)
        self.root_states[env_ids, :, 1:2] = 2.0 *torch_rand_float(-1., 1., (num_resets, self.num_robots, 1), self.device) + torch.tensor([0.], device=self.device)
        self.root_states[env_ids, :, 2:3] = .0 *torch_rand_float(-1., 1., (num_resets, self.num_robots, 1), self.device) + FLY_HEIGHT
        
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

        init_yaw = compute_direction_angle(self.root_states[env_ids, :, 0:2], self.goal_states[env_ids, :, 0:2], degrees=False).unsqueeze(-1)

        # randomize root orientation
        root_angle = torch.concatenate([0.*torch_rand_float(-torch.pi, torch.pi, (num_resets, self.num_robots, 2), self.device), # .01
                                        0.*torch_rand_float(-torch.pi, torch.pi, (num_resets, self.num_robots, 1), self.device) + init_yaw], dim=-1) # 0.05
        # root_angle = torch.concatenate([0.*torch_rand_float(-torch.pi, torch.pi, (num_resets, 2), self.device),# debug
        #                                 0.*torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device) + init_yaw], dim=-1) # debug

        matrix = T.euler_angles_to_matrix(root_angle, 'XYZ')
        root_quats = T.matrix_to_quaternion(matrix) # w,x,y,z
        self.root_states[env_ids, :, 3:7] = root_quats[:, :, [1, 2, 3, 0]] #x,y,z,w

        # randomize root linear and angular velocities
        self.root_states[env_ids, :, 7:10] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, self.num_robots, 3), self.device) # 0.5
        self.root_states[env_ids, :, 10:13] = 0.*torch_rand_float(-1.0, 1.0, (num_resets, self.num_robots, 3), self.device) # 0.2

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)

        self.reset_buf[env_ids] = 1
        self.progress_buf[env_ids] = 0

        self.pre_actions[env_ids] = 0
        self.prev_related_dist[env_ids] = 0
        self.pre_root_positions[env_ids] = 0
        self.pre_root_angvels[env_ids] = 0

        q_global = self.root_quats[..., [3, 0, 1, 2]]
        rot_matrix_global = T.quaternion_to_matrix(q_global)  # (num_envs, num_robots, 3, 3)

        yaw = torch.atan2(rot_matrix_global[..., 1, 0], rot_matrix_global[..., 0, 0])  # shape: (num_envs, num_robots)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)

        self.world_to_local = torch.stack([
            torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw,  cos_yaw, zeros], dim=-1),
            torch.stack([zeros,    zeros,   ones],  dim=-1),
        ], dim=-2)  # shape: (num_envs, num_robots, 3, 3)

        self.esdf_dist = torch.ones((self.num_envs, self.num_robots, 1), device=self.device) * 10
    
    def pre_physics_step(self, _actions):
        """
            _actions: <class 'torch.Tensor'> torch.Size([num_envs, num_robots, 4])  cpu     
        """
        self.counter += 1
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        self.actions = _actions.to(self.device) # [-1, 1]

        actions = self.actions
        if self.ctl_mode == 'rate' or self.ctl_mode == 'atti': 
            actions[..., -1] = 0.5 + 0.5 * self.actions[..., -1]
        actions = tensor_clamp(actions, self.action_lower_limits, self.action_upper_limits)

        actions_cpu = actions.cpu().numpy()

        #--------------- input state for pid controller. tensor [n,m,4] --------#
        obs_buf_cpu = self.root_states.cpu().numpy()
        for i in range(self.num_robots):
            # pos
            root_pos_cpu = self.root_states[..., i, 0:3].cpu().numpy()
            # quat. if w is negative, then set it to positive. x,y,z,w
            self.root_states[..., i, 3:7] = torch.where(self.root_states[..., i, 6:7] < 0, 
                                                 -self.root_states[..., i, 3:7], 
                                                 self.root_states[..., i, 3:7])
            root_quats_cpu = self.root_states[..., i, 3:7].cpu().numpy() # x,y,z,w
            # lin vel
            lin_vel_cpu = self.root_states[..., i, 7:10].cpu().numpy()
            # ang vel
            ang_vel_cpu = self.root_states[..., i, 10:13].cpu().numpy()

            control_mode_ = self.ctl_mode
            if(control_mode_ == "pos"):
                root_quats_cpu = root_quats_cpu[:, [3, 0, 1, 2]]
                self.parallel_pos_control[i].set_status(root_pos_cpu,root_quats_cpu,lin_vel_cpu,ang_vel_cpu,0.01)
                self.cmd_thrusts[:, i] = torch.tensor(self.parallel_pos_control[i].update(actions_cpu[:, i].astype(np.float64)))
            elif(control_mode_ == "vel"):
                root_quats_cpu = root_quats_cpu[:, [3, 0, 1, 2]]
                self.parallel_vel_control[i].set_status(root_pos_cpu,root_quats_cpu,lin_vel_cpu,ang_vel_cpu,0.01)
                self.cmd_thrusts[:, i] = torch.tensor(self.parallel_vel_control[i].update(actions_cpu[:, i].astype(np.float64)))
            elif(control_mode_ == "atti"):
                root_quats_cpu = root_quats_cpu[:, [3, 0, 1, 2]] # w, x, y, z
                self.parallel_atti_control[i].set_status(root_pos_cpu,root_quats_cpu,lin_vel_cpu,ang_vel_cpu,0.01)
                self.cmd_thrusts[:, i] = torch.tensor(self.parallel_atti_control[i].update(actions_cpu[:, i].astype(np.float64))) 
            elif(control_mode_ == "rate"):
                root_quats_cpu = root_quats_cpu[:, [3, 0, 1, 2]]
                self.parallel_rate_control[i].set_q_world(root_quats_cpu.astype(np.float64))
                # print("thrust", actions_cpu[0][-1])
                self.cmd_thrusts[:, i] = torch.tensor(self.parallel_rate_control[i].update(actions_cpu[:, i].astype(np.float64),ang_vel_cpu.astype(np.float64),0.01)) 
                # print("thrust on prop", self.cmd_thrusts[0])
            elif(control_mode_ == "prop"):
                self.cmd_thrusts[:, i] =  actions[:, i]
            else:
                print("Mode error")

            delta = .0*torch_rand_float(-1.0, 1.0, (self.num_envs, 1), device=self.device).repeat(1,4) + 9.59 
            thrusts=(self.cmd_thrusts[:, i, :].to(self.device) *delta)

            force_x = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
            force_y = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
            force_xy = torch.cat((force_x, force_y), 1).reshape(-1, 4, 2)
            thrusts = thrusts.reshape(-1, 4, 1)
            thrusts = torch.cat((force_xy, thrusts), 2)

            self.thrusts[:, i, :, :] = thrusts
            # clear actions for reset envs
            self.thrusts[reset_env_ids] = 0
            # spin spinning rotors
            prop_rot = ((self.cmd_thrusts[:, i, :])*0.2).to(self.device)
            self.torques[:, i*5+1, 2] = -prop_rot[:, 0]
            self.torques[:, i*5+2, 2] = -prop_rot[:, 1]
            self.torques[:, i*5+3, 2] = prop_rot[:, 2]
            self.torques[:, i*5+4, 2] = prop_rot[:, 3]

            self.forces[:, i*5+1:i*5+5, :] = self.thrusts[:, i, :, :]

        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(
                self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)
    
    def check_collisions(self):
        ones = torch.ones((self.num_envs), device=self.device)
        zeros = torch.zeros((self.num_envs), device=self.device)
        self.collisions[:] = 0
        for i in range(self.num_robots):
            # if any contact force is greater than 0.1, set collision to 1
            self.collisions[:, i] = torch.where(torch.norm(self.contact_forces[:, i, :], dim=-1) > 0.1, ones, zeros)
        self.all_collisions = torch.any(self.collisions > 0, dim=-1)

    def dump_images(self):
        for env_id in range(self.num_envs):
            for i in range(self.num_robots):
                # the depth values are in -ve z axis, so we need to flip it to positive
                self.full_camera_array[env_id, i, :] = -self.camera_tensors[env_id*self.num_robots+i].T
                self.full_camera_array[env_id, i, :] = torch.where(self.full_camera_array[env_id, i, :] > 4.5, torch.tensor(4.5), self.full_camera_array[env_id, i, :])
                self.full_camera_array[env_id, i, :] = torch.clamp(self.full_camera_array[env_id, i, :], 0, 4.5) / 4.5

                def add_gaussian_noise(depth_map, mean=0.0, std=.1):
                    noise = torch.normal(mean, std, size=depth_map.shape, device=depth_map.device)
                    noisy_depth_map = depth_map + noise
                    return torch.clamp(noisy_depth_map, 0.0, depth_map.max())
                
                def add_multiplicative_noise(depth_map, mean=1.0, std=0.3):
                    noise = torch.normal(mean, std, size=depth_map.shape, device=depth_map.device)
                    noisy_depth_map = depth_map * noise
                    return torch.clamp(noisy_depth_map, 0.0, depth_map.max())
                
                def apply_gaussian_blur(depth_map, kernel_size=5, sigma=1.0):
                    # Create a Gaussian kernel
                    kernel = torch.randint(0, 256, (kernel_size, kernel_size), dtype=torch.float32) / 256.0
                    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                    kernel = kernel.to(depth_map.device)

                    # Apply convolution for Gaussian blur
                    return F.conv2d(depth_map.unsqueeze(0), kernel, padding=kernel_size//2).squeeze(0)
                
                # self.full_camera_array[env_id, :] = add_gaussian_noise(self.full_camera_array[env_id, :])
                # self.full_camera_array[env_id, :] = add_multiplicative_noise(self.full_camera_array[env_id, :])
                # self.full_camera_array[env_id, :] = apply_gaussian_blur(self.full_camera_array[env_id, :])

                depth_image = self.full_camera_array[env_id, i, :].T.cpu().numpy()
                dist = cv2.normalize(depth_image, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                depth_colored = cv2.applyColorMap(dist, cv2.COLORMAP_PLASMA)
                # depth_colored = cv2.applyColorMap(dist, cv2.COLORMAP_JET)

                # cv2.imshow(str(env_id*self.num_robots+i), depth_colored)
                # cv2.waitKey(1)

    def step(self, actions):
        """
        step physics and render each frame. 
        """
        self.actions_local = actions = actions.view(self.num_envs, self.num_robots, self.num_actions).to(self.device)
        
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

        flattened_array = self.full_camera_array.clone().view(self.full_camera_array.size(0), self.full_camera_array.size(1), -1)
        self.esdf_dist = torch.min(flattened_array, dim=-1, keepdim=True).values
        self.compute_reward()

        if self.cfg.env.reset_on_collision:
            ones = torch.ones_like(self.reset_buf)
            self.reset_buf = torch.where(self.all_collisions > 0, ones, self.reset_buf)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.time_out_buf = self.progress_buf > self.max_episode_length
        self.extras["time_outs"] = self.time_out_buf.unsqueeze(0).expand(self.num_robots, -1).flatten()
        self.extras["item_reward_info"] = self.item_reward_info

        # reshape
        rsz_full_camera_array = self.full_camera_array.clone().view(-1, self.cam_channel, self.cam_resolution[0], self.cam_resolution[1])
        rsz_obs_buf = self.obs_buf.clone().view(-1, self.num_obs)
        rsz_rew_buf = self.rew_buf.clone().view(-1)
        rsz_reset_buf = self.reset_robot.clone().unsqueeze(0).flatten()

        obs = {
            'image': rsz_full_camera_array,
            'observation': rsz_obs_buf,
        }
        
        self.prev_related_dist = self.related_dist

        return obs, self.privileged_obs_buf, rsz_rew_buf, rsz_reset_buf, self.extras

    def compute_observations(self):
        forward_global = self.goal_positions - self.root_positions
        
        q_global = self.root_quats[..., [3, 0, 1, 2]]
        rot_matrix_global = T.quaternion_to_matrix(q_global)

        yaw = torch.atan2(rot_matrix_global[..., 1, 0], rot_matrix_global[..., 0, 0])  # (num_envs, num_robots)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        self.world_to_local = torch.stack([
            torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw,  cos_yaw, zeros], dim=-1),
            torch.stack([zeros,    zeros,   ones],  dim=-1),
        ], dim=-2)

        rot_matrix_local = torch.matmul(self.world_to_local, rot_matrix_global)
        self.euler_angles_local = T.matrix_to_euler_angles(rot_matrix_local, "XYZ")

        self.pos_diff_local = torch.einsum("bnij,bnj->bni", self.world_to_local, forward_global)
        self.vel_local       = torch.einsum("bnij,bnj->bni", self.world_to_local, self.root_linvels)
        self.ang_vel_local   = torch.einsum("bnij,bnj->bni", self.world_to_local, self.root_angvels)

        self.goal_dir = self.pos_diff_local / torch.norm(self.pos_diff_local, dim=-1, keepdim=True)  # (b, n, 3)
        self.related_dist = torch.norm(forward_global, dim=-1)  # (b, n)

        self.obs_buf[...,  0: 3] = self.goal_dir
        self.obs_buf[...,  3: 6] = self.euler_angles_local
        self.obs_buf[...,  6: 9] = self.vel_local
        self.obs_buf[...,  9:12] = self.ang_vel_local
        self.obs_buf[..., 12:16] = self.actions_local

        for idx in range(self.num_robots):
            pos_x = self.root_positions[:, idx, 0:1]
            vel_x = self.root_linvels[:, idx, 0:1]

            for j in range(self.num_robots):
                self.obs_buf[:, idx, 16+j*2:16+(j+1)*2-1] = self.root_positions[:, j, 0:1].clone() - pos_x.clone()
                self.obs_buf[:, idx, 16+(j+1)*2-1:16+(j+1)*2] = self.root_linvels[:, j, 0:1].clone() - vel_x.clone() 
        
        self.obs_buf[..., 16:] = 0
        
    def compute_reward(self):
        self.rew_buf[:], self.reset_robot[:], self.reset_buf[:], self.item_reward_info = self.compute_quadcopter_reward()
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
        forward_reward = .1 * (torch.norm(self.goal_positions - self.pre_root_positions, dim=-1) - torch.norm(self.goal_positions - self.root_positions, dim=-1))

        # heading reward
        forward_vec = self.pos_diff_local / torch.norm(self.pos_diff_local, dim=-1, keepdim=True)
        heading_vec = torch.tensor([1.0, 0.0, 0.0]).view(1, 1, -1).expand(self.num_envs, self.num_robots, -1).to(self.device)
        heading_reward = torch.sum(forward_vec * heading_vec, dim=-1)

        # speed reward
        speed_reward = -0.5 * (1 - torch.exp(-2 * torch.square(self.vel_local[..., 0] - 1.0)))

        # height reward
        z_reward = torch.min(torch.min(self.root_positions[..., 2] - (FLY_HEIGHT+0.3), torch.tensor(0.0)), (FLY_HEIGHT-0.3) - self.root_positions[..., 2])

        # ups reward
        q = self.root_quats.reshape(self.num_envs*self.num_robots, 4)
        ups = quat_axis(q, axis=2)
        ups = ups.view(self.num_envs, self.num_robots, 3)
        ups_reward = torch.square((ups[..., 2] + 1) / 2)

        # esdf reward
        esdf_reward = 0.5 * (1-torch.exp(- 0.5 * torch.square(self.esdf_dist))).squeeze(-1)

        # collision
        alive_reward = torch.where(self.esdf_dist > 0.3, torch.tensor(0.0), torch.tensor(-1.0)).squeeze(-1)

        # reach goal
        reach_goal = self.related_dist < 0.3
        # reach_goal_reward = torch.where(reach_goal, torch.tensor(20.0), torch.tensor(0.0))
        reach_goal_reward = torch.where(reach_goal, torch.tensor(200.0), torch.tensor(0.0))

        reward = (
            continous_action_reward
            + forward_reward
            + alive_reward + esdf_reward
            + ups_reward
            + z_reward
            + speed_reward
            + heading_reward
            + thrust_reward
            + reach_goal_reward
        )

        # resets due to misbehavior
        ones = torch.ones_like(self.reset_robot)
        die = torch.zeros_like(self.reset_robot)

        # resets due to too low or too high
        reset_robot = torch.where(self.root_positions[..., 2] > FLY_HEIGHT+0.3, ones, die)

        # resets due to collision or reach goal
        reset_robot = torch.where(self.collisions > 0, ones, reset_robot)
        reset_robot = torch.where(reach_goal, ones, reset_robot)

        # reset env
        reset_env = torch.any(reset_robot, dim=-1) # (num_envs,)
        # resets due to episode length
        reset_env = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), reset_env)

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
        
        return reward, reset_robot, reset_env, item_reward_info

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