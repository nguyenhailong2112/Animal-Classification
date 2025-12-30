import torch
import torch.nn as nn
from torchvision.models import resnet50

class SiameseNetwork(nn.Module):
    def __init__(self, pretrained_path):
        super(SiameseNetwork, self).__init__()
        self.backbone = resnet50(weights=None)
        checkpoint = torch.load(pretrained_path, weights_only=True)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )
        self.backbone.load_state_dict(checkpoint)
        self.projection = nn.Linear(512, 128)

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True

    def forward_one(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc[0](x)
        x = self.backbone.fc[1](x)
        x = self.backbone.fc[2](x)
        x = self.projection(x)
        return x

    def forward(self, img1, img2):
        feature1 = self.forward_one(img1)
        feature2 = self.forward_one(img2)
        distance = torch.nn.functional.pairwise_distance(feature1, feature2)
        return distance