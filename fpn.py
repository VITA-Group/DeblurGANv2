import torch
import torch.nn as nn

from torchvision.models import resnet50


class FPNSegHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


class FPNSeg(nn.Module):

    def __init__(self, num_classes, num_filters=128, num_filters_fpn=256, pretrained=True):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.

        self.fpn = FPN(num_filters=num_filters_fpn, pretrained=pretrained)

        # The segmentation heads on top of the FPN

        self.head1 = FPNSegHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNSegHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNSegHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNSegHead(num_filters_fpn, num_filters, num_filters)

        self.final = nn.Conv2d(4 * num_filters, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = self.head1(map1)

        final = self.final(torch.cat([map4, map3, map2, map1], dim=1))

        output = nn.functional.upsample(final, scale_factor=4, mode="nearest")

        return output


class FPN(nn.Module):

    def __init__(self, num_filters=256, pretrained=True):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()

        self.resnet = resnet50(pretrained=pretrained)

        self.enc0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.enc1 = self.resnet.layer1  # 256
        self.enc2 = self.resnet.layer2  # 512
        self.enc3 = self.resnet.layer3  # 1024
        self.enc4 = self.resnet.layer4  # 2048

        self.lateral4 = nn.Conv2d(2048, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(1024, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(512, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(256, num_filters, kernel_size=1, bias=False)

        self.smooth4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.smooth3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.smooth2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.smooth1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)


    def forward(self, x):

        # Bottom-up pathway, from ResNet

        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0) # 256
        enc2 = self.enc2(enc1) # 512
        enc3 = self.enc3(enc2) # 1024
        enc4 = self.enc4(enc3) # 2048

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)

        # Top-down pathway

        map4 = lateral4
        map3 = lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest")
        map2 = lateral2 + nn.functional.upsample(map3, scale_factor=2, mode="nearest")
        map1 = lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest")

        # Reduce aliasing effect of upsampling

        map4 = self.smooth4(map4)
        map3 = self.smooth3(map3)
        map2 = self.smooth2(map2)
        map1 = self.smooth1(map1)

        return map1, map2, map3, map4
