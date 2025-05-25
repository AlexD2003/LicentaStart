import torch
import torch.nn as nn
from torchvision import models

class ResNet18Classifier(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super(ResNet18Classifier, self).__init__()
        
        # Load pretrained ResNet-18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Modify the first conv layer to accept 1-channel input (grayscale)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Replace final FC layer with binary classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
