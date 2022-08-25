from collections import OrderedDict
import torch
import torch.nn as nn

from .backbones import DarkNet53, ResNet50, ConvNeXt
from configure import *


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups,
                           bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))


def conv_dw(filter_in, filter_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, (3, 3), stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),

        nn.Conv2d(filter_in, filter_out, (1, 1), 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )


#  SPP structure, using different size pooling kneral to do pooling.
#  Do concatation after pooling
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


class RFA(nn.Module):
    """
    Residual Feature Augmentation
    """
    def __init__(self, in_channel=768, neck_ratio=4, pooling_ratio=[1.0, 0.5, 0.4]):
        super(RFA, self).__init__()
        self.pooling_ratio = pooling_ratio
        self.in_channel = in_channel
        self.neck_ratio = neck_ratio

        self.conv1 = nn.ModuleList()
        self.conv1.extend([nn.Conv2d(self.in_channel, self.in_channel // self.neck_ratio, 1)
                           for _ in range(len(self.pooling_ratio))])
        self.activate = nn.GELU()
        self.conv2 = nn.Conv2d(self.in_channel // self.neck_ratio * len(self.pooling_ratio),
                               self.in_channel // self.neck_ratio, 1, padding=1)
        self.conv3 = nn.Conv2d(self.in_channel // self.neck_ratio, 512, 3)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        scaled_apf = []  # scalde adaptive pooling feature
        for i in range(len(self.pooling_ratio)):
            adaptive_pooling = nn.AdaptiveAvgPool2d((int(h * self.pooling_ratio[i]), int(w * self.pooling_ratio[i])))
            f = adaptive_pooling(x)
            f = self.conv1[i](f)
            # f = self.activate(f)
            upsample_bilinear2d = nn.UpsamplingBilinear2d(scale_factor=(1 / self.pooling_ratio[i]))
            f = upsample_bilinear2d(f)
            scaled_apf.append(f)
        concat_apf = torch.cat(scaled_apf, dim=1)
        P6 = self.conv2(concat_apf)
        P6 = self.conv3(P6)
        P6 = self.activate(P6)
        return P6


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, backbone="convnext_tiny", pool_ratios=None,
                 pretrained=False, residual_feature=RESIDUAL_FEATURE_AUG):
        super(YoloBody, self).__init__()

        # paras for residual feature augmentation
        if pool_ratios is None:
            self.pool_ratios = [0.1, 0.2, 0.3]
        self.res_feature = residual_feature

        if backbone == "resnet50":
            self.backbone = ResNet50(pretrained=pretrained)
            in_filters = [512, 1024, 2048]
        elif backbone == 'CSPDarknet53':
            self.backbone = DarkNet53(pretrained=pretrained)
            in_filters = [256, 512, 1024]
        elif backbone in ['convnext_tiny', 'convnext_small']:
            self.backbone = ConvNeXt(backbone, pretrained=pretrained)
            in_filters = [192, 384, 768]
        else:
            raise ValueError(
                'Unsupported backbone - `{}`, '
                'Use CSPDarknet53, resnet50, convnext_small, convnext_tiny.'.format(backbone))

        self.conv1 = make_three_conv([512, 1024], in_filters[2])
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = conv2d(in_filters[1], 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = conv2d(in_filters[0], 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        self.yolo_head3 = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)], 128)

        self.down_sample1 = conv_dw(128, 256, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)

        self.yolo_head2 = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)], 256)

        self.down_sample2 = conv_dw(256, 512, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)

        self.yolo_head1 = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)], 512)

        if self.res_feature:
            self.make_P6 = RFA(in_channel=in_filters[2])

    def forward(self, x):
        # x0 is C5
        x2, x1, x0 = self.backbone(x)

        P5 = self.conv1(x0)
        if self.res_feature:
            """
            creating P6 and add P6 to P5
            """
            P6 = self.make_P6(x0)
            """print("P6: ")
            print(P6.shape)
            print("P5: ")
            print(P5.shape)"""
            P5 += P6

        P5 = self.SPP(P5)
        P5 = self.conv2(P5)
        P5_upsample = self.upsample1(P5)

        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4, P5_upsample], axis=1)
        P4 = self.make_five_conv1(P4)
        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3, P4_upsample], axis=1)
        P3 = self.make_five_conv2(P3)
        P3_downsample = self.down_sample1(P3)

        P4 = torch.cat([P3_downsample, P4], axis=1)
        P4 = self.make_five_conv3(P4)
        P4_downsample = self.down_sample2(P4)

        P5 = torch.cat([P4_downsample, P5], axis=1)
        P5 = self.make_five_conv4(P5)

        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2
