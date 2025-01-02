import copy
import os

import numpy as np
import time
import gym

from datetime import datetime
from tensorboardX import SummaryWriter
import torch
from torch import nn
import torch.distributed as dist

from lib.core import common_losses
from lib.core.dignostics import DefaultDiagnostics, PpoDiagnostics
from lib.core.interval_summary_writer import IntervalSummaryWriter
from lib.core.experience import ExperienceBuffer, ExperienceBufferWithImage
from lib.core import schedulers
from lib.core import torch_ext
from lib.core.moving_mean_std import GeneralizedMovingStats
from lib.utils import vecenv

from abc import ABC
from abc import abstractmethod, abstractproperty

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

class BaseAlgorithm(ABC):
    def __init__(self, base_name, config):
        pass

    @abstractproperty
    def device(self):
        pass

    @abstractmethod
    def clear_stats(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def get_full_state_weights(self):
        pass

    @abstractmethod
    def set_full_state_weights(self, weights, set_epoch):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass

    # Get algo training parameters
    @abstractmethod
    def get_param(self, param_name):
        pass

    # Set algo training parameters
    @abstractmethod
    def set_param(self, param_name, param_value):
        pass

class A2CBase(BaseAlgorithm):

    def __init__(self, base_name, params):
        self.network_config = params['network']
        self.config = config = params['config']
        pbt_str = ''

        # generating a new name with the timestamp every time.
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")

        self.algo_observer = config['features']['observer']
        self.algo_observer.before_init(base_name, config, self.experiment_name)
        # self.load_networks(params) 

        self.multi_gpu = config.get('multi_gpu', False)

        # multi-gpu/multi-node data
        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        self.curr_frames = 0

        if self.multi_gpu:
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            dist.init_process_group("nccl", rank=self.global_rank, world_size=self.world_size)

            self.device_name = 'cuda:' + str(self.local_rank)
            config['device'] = self.device_name
            if self.global_rank != 0:
                config['print_stats'] = False
                config['lr_schedule'] = None

        self.use_diagnostics = config.get('use_diagnostics', False)

        if self.use_diagnostics and self.global_rank == 0:
            self.diagnostics = PpoDiagnostics()
        else:
            self.diagnostics = DefaultDiagnostics()

        self.network_path = config.get('network_path', "./nn/")
        self.log_path = config.get('log_path', "runs/")
        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']
        self.env_name = config['env_name']

        # self.env_info is environment config class cfg.env
        self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
        self.env_info = self.vec_env.get_env_info()

        self.ppo_device = config.get('device', 'cuda:0')
        self.value_size = self.env_info.get('value_size',1)
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.truncate_grads = self.config.get('truncate_grads', False)

        self.self_play_config = self.config.get('self_play_config', None)
        self.has_self_play_config = self.self_play_config is not None

        self.self_play = config.get('self_play', False)
        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.name = base_name

        self.ppo = config.get('ppo', True)
        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')

        # Setting learning rate scheduler
        if self.is_adaptive_lr:
            self.kl_threshold = config['kl_threshold']
            self.scheduler = schedulers.AdaptiveScheduler(self.kl_threshold)

        elif self.linear_lr:
            if self.max_epochs == -1 and self.max_frames == -1:
                print("Max epochs and max frames are not set. Linear learning rate schedule can't be used, switching to the contstant (identity) one.")
                self.scheduler = schedulers.IdentityScheduler()
            else:
                use_epochs = True
                max_steps = self.max_epochs

                if self.max_epochs == -1:
                    use_epochs = False
                    max_steps = self.max_frames

                self.scheduler = schedulers.LinearScheduler(float(config['learning_rate']), 
                    max_steps = max_steps,
                    use_epochs = use_epochs, 
                    apply_to_entropy = config.get('schedule_entropy', False),
                    start_entropy_coef = config.get('entropy_coef'))
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.horizon_length = config['horizon_length']

        self.normalize_advantage = config['normalize_advantage']
        self.normalize_rms_advantage = config.get('normalize_rms_advantage', False)
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k,v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
 
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.games_to_track = self.config.get('games_to_track', 100)
        print('current training device:', self.ppo_device)
        self.game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_shaped_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        self.obs = None

        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors

        assert(('minibatch_size_per_env' in self.config) or ('minibatch_size' in self.config))
        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)

        self.num_minibatches = self.batch_size // self.minibatch_size
        assert(self.batch_size % self.minibatch_size == 0)

        self.mini_epochs_num = self.config['mini_epochs']

        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0
        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.entropy_coef = self.config['entropy_coef']

        if self.global_rank == 0:
            writer = SummaryWriter(self.summaries_dir)
            self.writer = writer
        else:
            self.writer = None

        self.value_bootstrap = self.config.get('value_bootstrap')
        self.use_smooth_clamp = self.config.get('use_smooth_clamp', False)

        if self.use_smooth_clamp:
            self.actor_loss_func = common_losses.smoothed_actor_loss
        else:
            self.actor_loss_func = common_losses.actor_loss

        if self.normalize_advantage and self.normalize_rms_advantage:
            momentum = self.config.get('adv_rms_momentum', 0.5)
            self.advantage_mean_std = GeneralizedMovingStats((1,), momentum=momentum).to(self.ppo_device)

        self.is_tensor_obses = False

        self.last_state_indices = None

        # features
        self.algo_observer = config['features']['observer']

        self.soft_aug = config['features'].get('soft_augmentation', None)
        self.has_soft_aug = self.soft_aug is not None
        # soft augmentation not yet supported
        assert not self.has_soft_aug

    def trancate_gradients_and_step(self):
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))

            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.world_size
                    )
                    offset += param.numel()

        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?
        self.diagnostics.send_info(self.writer)
        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)

        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)
        self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
        self.writer.add_scalar('info/lr_mul', lr_mul, frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(kls).item(), frame)
        self.writer.add_scalar('info/epochs', epoch_num, frame)
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def set_eval(self):
        self.model.eval()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.train()

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)

        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            self.model.eval()
            processed_obs = self._preproc_obs(obs['obs'])
            input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : processed_obs,
            }
            result = self.model(input_dict)
            value = result['values']
            return value

    @property
    def device(self):
        return self.ppo_device

    def reset_envs(self):
        self.obs = self.env_reset()

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'use_action_masks' : self.use_action_masks
        }
        if 'resnet' in self.network_config:
            self.experience_buffer = ExperienceBufferWithImage(self.env_info, algo_info, self.ppo_device)
        else:
            self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        val_shape = (self.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.ppo_device)
            else:
                obs = torch.FloatTensor(obs).to(self.ppo_device)
        return obs

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def env_reset(self):
        obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)
        return obs

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    def discount_values_masks(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards, mb_masks):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            masks_t = mb_masks[t].unsqueeze(1)
            delta = (mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t])
            mb_advs[t] = lastgaelam = (delta + self.gamma * self.tau * nextnonterminal * lastgaelam) * masks_t
        return mb_advs

    def clear_stats(self):
        batch_size = self.num_agents * self.num_actors
        self.game_rewards.clear()
        self.game_shaped_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -100500
        self.algo_observer.after_clear_stats()

    def update_epoch(self):
        pass

    def train(self):
        pass

    def prepare_dataset(self, batch_dict):
        pass

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

    def train_actor_critic(self, obs_dict, opt_step=True):
        pass

    def calc_gradients(self):
        pass

    def get_central_value(self, obs_dict):
        return self.central_value_net.get_value(obs_dict)

    def train_central_value(self):
        return self.central_value_net.train_net()

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        state['frame'] = self.frame
        state['optimizer'] = self.optimizer.state_dict()

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards'] = self.last_mean_rewards

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        return state

    def set_full_state_weights(self, weights, set_epoch=True):

        self.set_weights(weights)
        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        self.optimizer.load_state_dict(weights['optimizer'])

        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    def get_weights(self):
        state = self.get_stats_weights()
        state['model'] = self.model.state_dict()
        return state

    def get_stats_weights(self, model_stats=False):
        state = {}
        if self.mixed_precision:
            state['scaler'] = self.scaler.state_dict()
        if model_stats:
            if self.normalize_input:
                state['running_mean_std'] = self.model.running_mean_std.state_dict()
            if self.normalize_value:
                state['reward_mean_std'] = self.model.value_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        if self.normalize_rms_advantage:
            self.advantage_mean_std.load_state_dic(weights['advantage_mean_std'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value and 'normalize_value' in weights:
            self.model.value_mean_std.load_state_dict(weights['reward_mean_std'])
        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        self.set_stats_weights(weights)

    def get_param(self, param_name):
        if param_name in [
            "grad_norm",
            "critic_coef", 
            "bounds_loss_coef",
            "entropy_coef",
            "kl_threshold",
            "gamma",
            "tau",
            "mini_epochs_num",
            "e_clip",
            ]:
            return getattr(self, param_name)
        elif param_name == "learning_rate":
            return self.last_lr
        else:
            raise NotImplementedError(f"Can't get param {param_name}")       

    def set_param(self, param_name, param_value):
        if param_name in [
            "grad_norm",
            "critic_coef", 
            "bounds_loss_coef",
            "entropy_coef",
            "gamma",
            "tau",
            "mini_epochs_num",
            "e_clip",
            ]:
            setattr(self, param_name, param_value)
        elif param_name == "learning_rate":
            if self.global_rank == 0:
                if self.is_adaptive_lr:
                    raise NotImplementedError("Can't directly mutate LR on this schedule")
                else:
                    self.learning_rate = param_value

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
        elif param_name == "kl_threshold":
            if self.global_rank == 0:
                if self.is_adaptive_lr:
                    self.kl_threshold = param_value
                    self.scheduler.kl_threshold = param_value
                else:
                    raise NotImplementedError("Can't directly mutate kl threshold")
        else:
            raise NotImplementedError(f"No param found for {param_value}")

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k,v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]
     
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict
