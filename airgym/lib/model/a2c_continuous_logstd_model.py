import torch
import numpy as np
import torch.nn as nn
from airgym.lib.model.base_model import BaseModel
from airgym.lib.network.mlp import MLP
from airgym.lib.network.resnet import ResNetFeatureExtractor
from airgym.lib.network.cnn import CNNFeatureExtractor
from airgym.lib.core.running_mean_std import RunningMeanStd, RunningMeanStdObs

class ModelA2CContinuousLogStd(BaseModel):
    def __init__(self, params, keys):
        super(BaseModel, self).__init__()
        actions_num = keys.get('actions_num')
        input_shape = keys.get('input_shape')
        self.normalize_value = params["config"].get('normalize_value', False)
        self.normalize_input = params["config"].get('normalize_input', False)
        self.value_size = params["config"].get('value_size', 1)
        self.num_seqs = num_seqs = keys.get('num_seqs', 1)

        self.load(params['network'])

        if self.has_resnet:
            self.actor_resnet = ResNetFeatureExtractor(self.resnet_type, output_dim=self.feature_dim)
            self.actor_mlp = MLP(input_shape['observation'][0]+self.feature_dim, self.mlp_cfg['units'], self.mlp_cfg['activation'])
        elif self.has_cnn:
            self.actor_cnn = CNNFeatureExtractor(feature_dim=self.feature_dim)
            self.actor_mlp = MLP(input_shape['observation'][0]+self.feature_dim, self.mlp_cfg['units'], self.mlp_cfg['activation'])
        else:
            self.actor_mlp = MLP(input_shape[0], self.mlp_cfg['units'], self.mlp_cfg['activation'])
        
        if self.separate:
            if self.has_resnet:
                self.critic_resnet = ResNetFeatureExtractor(self.resnet_type, output_dim=self.feature_dim)
                self.critic_mlp = MLP(input_shape['observation'][0]+self.feature_dim, self.mlp_cfg['units'], self.mlp_cfg['activation'])
            elif self.has_cnn:
                self.critic_cnn = CNNFeatureExtractor(feature_dim=self.feature_dim)
                self.critic_mlp = MLP(input_shape['observation'][0]+self.feature_dim, self.mlp_cfg['units'], self.mlp_cfg['activation'])
            else:
                self.critic_mlp = MLP(input_shape[0], self.mlp_cfg['units'], self.mlp_cfg['activation'])

        out_size = self.mlp_cfg['units'][-1]
        self.mu = nn.Linear(out_size, actions_num)
        self.mu_act = nn.Identity()
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        
        if self.fixed_sigma:
            self.logstd = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
            nn.init.constant_(self.logstd, 0.)
        else:
            self.logstd = torch.nn.Linear(out_size, actions_num)
            nn.init.constant_(self.logstd.weight, 0.)
        self.logstd_act = nn.Identity()

        self.value_head = torch.nn.Linear(self.mlp_cfg['units'][-1], 1)
        self.value_head_act = nn.Identity()
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,)) #   GeneralizedMovingStats((self.value_size,)) #   
        if self.normalize_input:
            if isinstance(input_shape, dict):
                # add feature_dim to input_shape["observation"]
                input_shape["observation"] = (input_shape["observation"][0] + self.feature_dim,)
                self.running_mean_std = RunningMeanStdObs(input_shape)
            else:
                self.running_mean_std = RunningMeanStd(input_shape)

        # for param in self.actor_mlp.parameters():
        #     param.requires_grad = False
        # for param in self.mu.parameters():
        #     param.requires_grad = False
        # self.logstd.requires_grad = False

    def forward(self, input_dict):
        is_train = input_dict.get('is_train', True)
        prev_actions = input_dict.get('prev_actions', None)
        # norm_out = self.norm_obs(input_dict['obs'])
        
        if self.separate:
            if self.has_resnet:
                normed_image = self.norm_image(input_dict['obs']['image'])
                a_resnet_out = self.actor_resnet(normed_image)
                c_resnet_out = self.critic_resnet(normed_image)
                
                a_out = torch.cat((input_dict['obs']['observation'], a_resnet_out), dim=-1)
                c_out = torch.cat((input_dict['obs']['observation'], c_resnet_out), dim=-1)

                normed_a_out = self.norm_observation(a_out)
                normed_c_out = self.norm_observation(c_out)

                a_out = self.actor_mlp(normed_a_out)
                c_out = self.critic_mlp(normed_c_out)

            elif self.has_cnn:
                normed_image = self.norm_image(input_dict['obs']['image'])
                a_cnn_out = self.actor_cnn(normed_image)
                c_cnn_out = self.critic_cnn(normed_image)
                
                a_out = torch.cat((input_dict['obs']['observation'], a_cnn_out), dim=-1)
                c_out = torch.cat((input_dict['obs']['observation'], c_cnn_out), dim=-1)

                normed_a_out = self.norm_observation(a_out)
                normed_c_out = self.norm_observation(c_out)

                a_out = self.actor_mlp(normed_a_out)
                c_out = self.critic_mlp(normed_c_out)

            else:
                norm_out = self.norm_obs(input_dict['obs'])
                a_out = self.actor_mlp(norm_out)
                c_out = self.critic_mlp(norm_out)

            mu = self.mu_act(self.mu(a_out))
            if self.fixed_sigma:
                logstd = mu * 0.0 + self.logstd_act(self.logstd)
            else:
                logstd = self.logstd_act(self.logstd(a_out))
            value = self.value_head_act(self.value_head(c_out))

        else:
            if self.has_resnet:
                normed_image = self.norm_image(input_dict['obs']['image'])
                a_resnet_out = c_resnet_out = self.actor_resnet(normed_image)
                out = torch.cat((input_dict['obs']['observation'], a_resnet_out), dim=-1)
                norm_out = self.norm_observation(out)
            elif self.has_cnn:
                normed_image = self.norm_image(input_dict['obs']['image'])
                a_cnn_out = c_cnn_out = self.actor_cnn(normed_image)
                out = torch.cat((input_dict['obs']['observation'], a_cnn_out), dim=-1)
                norm_out = self.norm_observation(out)
            else:
                norm_out = self.norm_obs(input_dict['obs'])
            
            a_out = c_out = self.actor_mlp(norm_out)
            mu = self.mu_act(self.mu(a_out))
            if self.fixed_sigma:
                logstd = mu * 0.0 + self.logstd_act(self.logstd)
            else:
                logstd = self.logstd_act(self.logstd(a_out))
            value = self.value_head_act(self.value_head(c_out))
                
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma, validate_args=False)
        if is_train:
            entropy = distr.entropy().sum(dim=-1)
            prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
            result = {
                'prev_neglogp' : torch.squeeze(prev_neglogp),
                'values' : value,
                'entropy' : entropy,
                'mus' : mu,
                'sigmas' : sigma
            }
            return result
        else:
            selected_action = distr.sample()
            neglogp = self.neglogp(selected_action, mu, sigma, logstd)
            result = {
                'neglogpacs' : torch.squeeze(neglogp),
                'values' : self.denorm_value(value),
                'actions' : selected_action,
                'mus' : mu,
                'sigmas' : sigma
            }
            return result
    
    def neglogp(self, x, mean, std, logstd):
        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
            + logstd.sum(dim=-1)
        
    def load(self, params):
        self.separate = params.get('separate', False)
        self.mlp_cfg = params['mlp']
        self.has_space = 'space' in params
        self.has_resnet = 'resnet' in params
        self.has_cnn = 'cnn' in params

        if self.has_space:
            self.is_continuous = 'continuous'in params['space']
            if self.is_continuous:
                self.space_config = params['space']['continuous']
                self.fixed_sigma = self.space_config['fixed_sigma']
        else:
            self.is_discrete = False
            self.is_continuous = False
            self.is_multi_discrete = False
        
        if self.has_resnet:
            self.resnet_type = params['resnet']['type']
            self.feature_dim = params['resnet']['output_dim']
        
        if self.has_cnn:
            self.feature_dim = params['cnn']['output_dim']