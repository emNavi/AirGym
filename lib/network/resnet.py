import torch.nn as nn
from torchvision.models import resnet18

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, type, output_dim):
        super(ResNetFeatureExtractor, self).__init__()
        if type == 'resnet18':
            self.resnet = resnet18(pretrained=False)
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x):
        return self.resnet(x)