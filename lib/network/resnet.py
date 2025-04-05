import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, type, output_dim):
        super(ResNetFeatureExtractor, self).__init__()
        if type == 'resnet18':
            self.resnet = resnet18(pretrained=True)
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            for param in self.resnet.parameters():
                param.requires_grad = False
            
            in_features = self.resnet.fc.in_features  # 获取 fc 层输入维度 (512)
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # 去掉全连接层

            self.fc = nn.Linear(in_features, output_dim)


    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x