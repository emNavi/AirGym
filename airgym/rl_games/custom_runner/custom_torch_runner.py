from rl_games.torch_runner import Runner
from airgym.rl_games.algos_torch.tcn_a2c_continuous import TCNA2CAgent, TCNPlayerContinuous
from airgym.rl_games.networks.resnet import ResNetMLPBuilder

class CustomRunner(Runner):
    def __init__(self, algo_observer=None):
        super().__init__(algo_observer)
        # for TCN A2C training and playing
        self.algo_factory.register_builder('tcn_a2c_continuous', lambda **kwargs : TCNA2CAgent(**kwargs))
        self.player_factory.register_builder('tcn_a2c_continuous', lambda **kwargs : TCNPlayerContinuous(**kwargs))

        self.algo_factory.register_builder('resnet_mlp_a2c_continuous', lambda **kwargs : TCNA2CAgent(**kwargs))