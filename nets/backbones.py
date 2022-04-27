import torch.nn as nn
import torch.nn.functional as F

from .densenet import _Transition, densenet121, densenet169, densenet201
from .mobilenet_v1 import mobilenet_v1
from .mobilenet_v2 import mobilenet_v2
from .mobilenet_v3 import mobilenet_v3
from .resnet import resnet50
from .CSPdarknet import darknet53
from .convnext import convnext_tiny, convnext_small


class MobileNetV1(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV1, self).__init__()
        self.model = mobilenet_v1(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.stage1(x)
        out4 = self.model.stage2(out3)
        out5 = self.model.stage3(out4)
        return out3, out4, out5


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:14](out3)
        out5 = self.model.features[14:18](out4)
        return out3, out4, out5


class MobileNetV3(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV3, self).__init__()
        self.model = mobilenet_v3(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:13](out3)
        out5 = self.model.features[13:16](out4)
        return out3, out4, out5


class Densenet(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(Densenet, self).__init__()
        densenet = {
            "densenet121": densenet121,
            "densenet169": densenet169,
            "densenet201": densenet201
        }[backbone]
        model = densenet(pretrained)
        del model.classifier
        self.model = model

    def forward(self, x):
        feature_maps = []
        for block in self.model.features:
            if type(block) == _Transition:
                for _, subblock in enumerate(block):
                    x = subblock(x)
                    if type(subblock) == nn.Conv2d:
                        feature_maps.append(x)
            else:
                x = block(x)
        x = F.relu(x, inplace=True)
        feature_maps.append(x)
        return feature_maps[1:]


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
