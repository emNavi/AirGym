from rl_games.torch_runner import Runner
from airgym.rl_games.algos_torch.custom_a2c_continuous import CustomA2CAgent, CustomPlayerContinuous

class CustomRunner(Runner):
    def __init__(self, algo_observer=None):
        super().__init__(algo_observer)
        # for TCN A2C training and playing
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : CustomA2CAgent(**kwargs))
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : CustomPlayerContinuous(**kwargs))