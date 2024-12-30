import torch
from torch import nn
from torchvision.models import resnet18

from rl_games.algos_torch.network_builder import NetworkBuilder

class ResNetMLP(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.value_size = kwargs.pop('value_size', 1)

        NetworkBuilder.BaseNetwork.__init__(self)

        self.load(params)
        
        # Load pre-trained ResNet and modify it
        base_resnet = resnet18(pretrained=False)
        base_resnet.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(base_resnet.children())[:-1])  # Exclude the final FC layer
        
        resnet_out_dim = base_resnet.fc.in_features

        # MLP for 18-dimensional state vector
        state_input_dim = 18
        state_mlp_args = {
            'input_size': state_input_dim,
            'units': self.state_mlp_units,
            'activation': self.activation,
            'norm_func_name': self.normalization,
            'dense_func': torch.nn.Linear,
            'd2rl': self.is_d2rl,
            'norm_only_first_layer': self.norm_only_first_layer
        }
        self.state_mlp = self._build_mlp(**state_mlp_args)
        state_out_dim = self.state_mlp_units[-1] if self.state_mlp_units else state_input_dim

        # Combine ResNet and state MLP outputs
        combined_input_dim = resnet_out_dim + state_out_dim
        combined_mlp_args = {
            'input_size': combined_input_dim,
            'units': self.units,
            'activation': self.activation,
            'norm_func_name': self.normalization,
            'dense_func': torch.nn.Linear,
            'd2rl': self.is_d2rl,
            'norm_only_first_layer': self.norm_only_first_layer
        }
        self.actor_mlp = self._build_mlp(**combined_mlp_args)

        if self.separate:
            self.critic_mlp = self._build_mlp(**combined_mlp_args)

        out_size = self.units[-1] if self.units else combined_input_dim

        # Define output layers
        self.value = self._build_value_layer(out_size, self.value_size)
        self.value_act = self.activations_factory.create(self.value_activation)

        if self.is_discrete:
            self.logits = torch.nn.Linear(out_size, actions_num)
        elif self.is_multi_discrete:
            self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
        elif self.is_continuous:
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
            self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])

            if self.fixed_sigma:
                self.sigma = nn.Parameter(torch.zeros(actions_num, dtype=torch.float32), requires_grad=True)
            else:
                self.sigma = torch.nn.Linear(out_size, actions_num)

    def forward(self, obs_dict):
        depth_obs = obs_dict['depth']  # Depth image
        state_obs = obs_dict['state']  # 18-dimensional state vector
        states = obs_dict.get('rnn_states', None)

        # Process depth image through ResNet
        depth_features = self.resnet(depth_obs).flatten(1)

        # Process state vector through MLP
        state_features = self.state_mlp(state_obs)

        # Concatenate features
        combined_features = torch.cat([depth_features, state_features], dim=1)

        if self.separate:
            actor_out = self.actor_mlp(combined_features)
            critic_out = self.critic_mlp(combined_features)

            value = self.value_act(self.value(critic_out))

            if self.is_discrete:
                logits = self.logits(actor_out)
                return logits, value, states

            if self.is_multi_discrete:
                logits = [logit(actor_out) for logit in self.logits]
                return logits, value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(actor_out))
                sigma = self.sigma_act(self.sigma(actor_out)) if not self.fixed_sigma else self.sigma
                return mu, sigma, value, states

        else:
            actor_out = self.actor_mlp(combined_features)
            value = self.value_act(self.value(actor_out))

            if self.is_discrete:
                logits = self.logits(actor_out)
                return logits, value, states

            if self.is_multi_discrete:
                logits = [logit(actor_out) for logit in self.logits]
                return logits, value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(actor_out))
                sigma = self.sigma_act(self.sigma(actor_out)) if not self.fixed_sigma else self.sigma
                return mu, sigma, value, states

    def load(self, params):
        self.separate = params.get('separate', False)
        self.units = params['mlp']['units']
        self.activation = params['mlp']['activation']
        self.normalization = params.get('normalization', None)
        self.value_activation = params.get('value_activation', 'None')
        self.is_d2rl = params['mlp'].get('d2rl', False)
        self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)

        self.is_discrete = params['space'].get('discrete', False)
        self.is_multi_discrete = params['space'].get('multi_discrete', False)
        self.is_continuous = params['space'].get('continuous', False)

        if self.is_continuous:
            self.space_config = params['space']['continuous']
            self.fixed_sigma = self.space_config['fixed_sigma']
        elif self.is_discrete:
            self.space_config = params['space']['discrete']
        elif self.is_multi_discrete:
            self.space_config = params['space']['multi_discrete']

        self.state_mlp_units = params['state_mlp']['units']

class ResNetMLPBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return ResNetMLP(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)
