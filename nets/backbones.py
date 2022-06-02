import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50
from .CSPdarknet import darknet53
from .convnext import convnext_tiny, convnext_small


class ResNet50(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet50, self).__init__()
        self.model = resnet50(pretrained)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        feat1 = self.model.relu(x)

        x = self.model.maxpool(feat1)
        feat2 = self.model.layer1(x)

        feat3 = self.model.layer2(feat2)
        feat4 = self.model.layer3(feat3)
        feat5 = self.model.layer4(feat4)
        return [feat3, feat4, feat5]


class ConvNeXt(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(ConvNeXt, self).__init__()
        convnext = {
            'convnext_small': convnext_small,
            'convnext_tiny': convnext_tiny
        }[backbone]
        self.model = convnext(pretrained)

    def forward(self, x):
        x = self.model.forward(x)
        return x[1:]


class DarkNet53(nn.Module):
    def __init__(self, pretrained=False):
        super(DarkNet53, self).__init__()
        self.model = darknet53(pretrained=pretrained)

    def forward(self, x):
        x = self.model.forward(x)
        return x
