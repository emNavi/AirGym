import torch
from torch import nn
import torch.nn.functional as F

from rl_games.algos_torch.network_builder import NetworkBuilder

class TCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=(kernel_size-1)*dilation//2, dilation=dilation)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, padding=(kernel_size-1)*dilation//2, dilation=dilation)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.upsample = nn.Conv1d(input_dim, output_dim, kernel_size=1) if input_dim != output_dim else None

    def forward(self, x):
        # 输入形状为 (batch_size, input_dim, seq_length)
        out = self.conv1(x)
        out = F.relu(self.layer_norm1(out.permute(0, 2, 1)).permute(0, 2, 1))  # 转换维度以适配 LayerNorm
        out = self.conv2(out)
        out = F.relu(self.layer_norm2(out.permute(0, 2, 1)).permute(0, 2, 1))
        if self.upsample is not None:
            residual = self.upsample(x)
        else:
            residual = x
        out = out + residual  # 残差连接
        return out
    
class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size, num_layers):
        super(TCN, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # 指数增长的扩张率
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(TCNBlock(in_dim, hidden_dim, kernel_size, dilation))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 输入形状为 (batch_size, seq_length, input_dim)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_dim, seq_length)
        x = self.network(x)
        x = x.mean(dim=-1)  # 跨时间轴求平均
        x = self.fc(x)
        return x

class A2CTCN(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.value_size = kwargs.pop('value_size', 1)

        NetworkBuilder.BaseNetwork.__init__(self)

        self.load(params)
        self.actor_tcn = nn.Sequential()
        self.actor_mlp = nn.Sequential()
        self.critic_tcn = nn.Sequential()
        self.critic_mlp = nn.Sequential()

        tcn_args = {
            'input_dim' : input_shape[1],
            'hidden_dim' : self.tcn_hidden_dim,
            'output_dim' : self.tcn_output_dim,
            'kernel_size' : self.tcn_kernel_size,
            'num_layers' : self.tcn_num_layers,
        }
        
        self.actor_tcn = TCN(**tcn_args)
        if self.separate:
            self.critic_tcn = TCN(**tcn_args)

        in_mlp_shape = self.tcn_output_dim
        if len(self.units) == 0:
            out_size = in_mlp_shape
        else:
            out_size = self.units[-1]
        mlp_args = {
            'input_size' : in_mlp_shape, 
            'units' : self.units, 
            'activation' : self.activation, 
            'norm_func_name' : self.normalization,
            'dense_func' : torch.nn.Linear,
            'd2rl' : self.is_d2rl,
            'norm_only_first_layer' : self.norm_only_first_layer
        }
        self.actor_mlp = self._build_mlp(**mlp_args)
        
        if self.separate:
            self.critic_mlp = self._build_mlp(**mlp_args)

        self.value = self._build_value_layer(out_size, self.value_size)
        self.value_act = self.activations_factory.create(self.value_activation)

        if self.is_discrete:
            self.logits = torch.nn.Linear(out_size, actions_num)
        '''
            for multidiscrete actions num is a tuple
        '''
        if self.is_multi_discrete:
            self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
        if self.is_continuous:
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
            sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

            if self.fixed_sigma:
                self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
            else:
                self.sigma = torch.nn.Linear(out_size, actions_num)
        
        mlp_init = self.init_factory.create(**self.initializer)
        if self.has_tcn:
            tcn_init = self.init_factory.create(**self.tcn['initializer'])

        for m in self.modules():         
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                tcn_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        if self.is_continuous:
            mu_init(self.mu.weight)
            if self.fixed_sigma:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)
        

    def forward(self, obs_dict):
        obs = obs_dict['obs']
        states = obs_dict.get('rnn_states', None)
        dones = obs_dict.get('dones', None)
        bptt_len = obs_dict.get('bptt_len', 0)

        if self.separate:
            a_out = c_out = obs
            a_out = self.actor_tcn(a_out)
            c_out = self.critic_tcn(c_out)

            a_out = self.actor_mlp(a_out)
            c_out = self.critic_mlp(c_out)

            value = self.value_act(self.value(c_out))

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits, value, states

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits, value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.fixed_sigma:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma, value, states
        else:
            out = obs
            out = self.actor_tcn(out)
            out = self.actor_mlp(out)

            value = self.value_act(self.value(out))

            if self.central_value:
                return value, states

            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, states
            if self.is_multi_discrete:
                logits = [logit(out) for logit in self.logits]
                return logits, value, states
            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu*0 + sigma, value, states

    def is_separate_critic(self):
        return self.separate
        
    def load(self, params):
        self.separate = params.get('separate', False)
        self.units = params['mlp']['units']
        self.activation = params['mlp']['activation']
        self.initializer = params['mlp']['initializer']
        self.is_d2rl = params['mlp'].get('d2rl', False)
        self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
        self.value_activation = params.get('value_activation', 'None')
        self.normalization = params.get('normalization', None)
        self.has_tcn = 'tcn' in params
        self.has_space = 'space' in params
        self.central_value = params.get('central_value', False)
        self.joint_obs_actions_config = params.get('joint_obs_actions', None)

        if self.has_space:
            self.is_multi_discrete = 'multi_discrete'in params['space']
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

        if self.has_tcn:
            self.tcn = params['tcn']
            self.tcn_hidden_dim = params['tcn']['hidden_dim']
            self.tcn_output_dim = params['tcn']['output_dim']
            self.tcn_num_layers = params['tcn'].get('num_layers', 2)
            self.tcn_kernel_size = params['tcn'].get('kernel_size', 3)


class TCNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return A2CTCN(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)