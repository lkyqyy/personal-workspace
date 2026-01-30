import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    两个连续的卷积 + ReLU
    结构: Conv2d -> ReLU -> Conv2d -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """
    下采样模块: MaxPool2d(2) + DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    上采样模块: 上采样 + concat skip + DoubleConv，为了部署友好，使用双线性插值上采样 + 普通卷积，避免转置卷积和复杂的动态 padding。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        """
        x: 来自下一级 decoder 的特征（需上采样）
        skip: 对应 encoder 的特征（做 skip connection）
        """
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """输出层: 1x1 卷积，把通道数压到 num_classes"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    简单 UNet 结构，适配 256x256 输入
    encoder: 64 -> 128 -> 256 -> 512
    bottleneck: 1024
    decoder: 512 -> 256 -> 128 -> 64
    最终输出通道数 = num_classes（这里是 1）
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 8)

        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2 + base_channels, base_channels)

        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_center = self.bottleneck(x5)

        x = self.up1(x_center, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits
