import torch.nn as nn
import torch
from lib.core.running_mean_std import RunningMeanStd, RunningMeanStdObs

class BaseModel(nn.Module):
    def __init__(self, input_shape, config):
        super(BaseModel, self).__init__()
        self.obs_shape = input_shape
        self.normalize_value = config.get('normalize_value', False)
        self.normalize_input = config.get('normalize_input', False)
        self.value_size = config.get('value_size', 1)

        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,)) #   GeneralizedMovingStats((self.value_size,)) #   
        if self.normalize_input:
            if isinstance(self.obs_shape, dict):
                self.running_mean_std = RunningMeanStdObs(self.obs_shape)
            else:
                self.running_mean_std = RunningMeanStd(self.obs_shape)
    
    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation) if self.normalize_input else observation
        
    def norm_image(self, image):
        with torch.no_grad():
            return self.running_mean_std.running_mean_std["image"](image) if self.normalize_input else image
        
    def norm_observation(self, observation):
        with torch.no_grad():
            return self.running_mean_std.running_mean_std["observation"](observation) if self.normalize_input else observation

    def denorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, denorm=True) if self.normalize_value else value
                