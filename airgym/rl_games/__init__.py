from airgym.rl_games.networks.tcn import TCNetBuilder
from rl_games.algos_torch import model_builder
model_builder.register_network('tcn', TCNetBuilder)