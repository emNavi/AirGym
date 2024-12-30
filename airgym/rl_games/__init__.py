from airgym.rl_games.networks.tcn import TCNetBuilder
from airgym.rl_games.networks.resnet import ResNetMLPBuilder

from rl_games.algos_torch import model_builder, network_builder

model_builder.register_network('tcn', TCNetBuilder)
# network_builder.register_builder('tcn', TCNetBuilder)