import torch
import torch.nn as nn


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class CTS(torch.nn.Module):
    def __init__(self, out_planes, heads):
        super(CTS, self).__init__()
        self.out_planes = out_planes
        self.conv_1 = conv3x3(self.out_planes, self.out_planes, stride=1)
        self.attention = nn.MultiheadAttention(embed_dim=self.out_planes, num_heads=heads)
        self.bn = nn.BatchNorm2d(self.out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv3x3(self.out_planes, self.out_planes, stride=1)

    def forward(self, x):
        identity = x
        x = self.conv_1(x)
        output, _ = self.attention(x.view(x.size(0), x.size(2) * x.size(3), -1),
                                   x.view(x.size(0), x.size(2) * x.size(3), -1),
                                   x.view(x.size(0), x.size(2) * x.size(3), -1))
        output = output.view(x.size())
        output = self.relu(self.bn(output) + identity)
        output_ = self.conv(output)
        output_ = self.bn(output_)
        return self.relu(output_ + output)
