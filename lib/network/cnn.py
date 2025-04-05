import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=12):
        super(CNNFeatureExtractor, self).__init__()
        
        self.features = nn.Sequential(
            # input: (1, 212, 120)
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # (16, 106, 60)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # (32, 53, 30)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # (64, 27, 15)
            
            nn.AdaptiveAvgPool2d((1, 1)) 
            # (64, 1, 1)
        )
        
        self.fc = nn.Linear(64, feature_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x