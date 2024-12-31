import torch
import numpy as np
import torch.nn as nn
from lib.model.base_model import BaseModel
from lib.network.mlp import MLP

class ModelA2CContinuousLogStd(BaseModel):
    def __init__(self, params, keys):
        actions_num = keys.get('actions_num')
        input_shape = keys.get('input_shape')
        self.value_size = keys.get('value_size', 1)
        self.num_seqs = num_seqs = keys.get('num_seqs', 1)
        super().__init__(input_shape, params['config'])

        self.load(params['network'])

        self.actor_mlp = MLP(input_shape[0], self.mlp_cfg['units'], self.mlp_cfg['activation'])
        if self.separate:
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

    def forward(self, input_dict):
        is_train = input_dict.get('is_train', True)
        prev_actions = input_dict.get('prev_actions', None)
        out = self.norm_obs(input_dict['obs'])

        a_out = c_out = self.actor_mlp(out)
        mu = self.mu_act(self.mu(a_out))
        if self.fixed_sigma:
            logstd = mu * 0.0 + self.logstd_act(self.logstd)
        else:
            logstd = self.logstd_act(self.logstd(a_out))
        
        if self.separate:
            c_out = self.critic_mlp(out)
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
        if 'cnn' in params:
            self.resnet_cfg = params['resnet']
        self.value_activation = params.get('value_activation', 'None')
        self.normalization = params.get('normalization', None)
        self.has_space = 'space' in params
        self.central_value = params.get('central_value', False)
        self.joint_obs_actions_config = params.get('joint_obs_actions', None)

        if self.has_space:
            self.is_multi_discrete = 'multi_discrete' in params['space']
            self.is_discrete = 'discrete' in params['space']
            self.is_continuous = 'continuous'in params['space']
            if self.is_continuous:
                self.space_config = params['space']['continuous']
                self.fixed_sigma = self.space_config['fixed_sigma']
            elif self.is_discrete:
                self.space_config = params['space']['discrete']
            elif self.is_multi_discrete:
                self.space_config = params['space']['multi_discrete']
        else:
            self.is_discrete = False
            self.is_continuous = False
            self.is_multi_discrete = False