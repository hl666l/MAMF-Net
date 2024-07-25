import torch
import torch.nn as nn
from Tools.RSCA import RSCA
from Tools.CTS import CTS
from Base_Block import Bottleneck


class MAMF(nn.Module):

    def __init__(self):
        super(MAMF, self).__init__()
        # 卷积网络
        self.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, 1, True),
            Bottleneck(256, 64, 1, False),
            Bottleneck(256, 64, 1, False),
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, 2, True),
            Bottleneck(512, 128, 1, False),
            Bottleneck(512, 128, 1, False),
            Bottleneck(512, 128, 1, False)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(512, 256, 2, True),
            Bottleneck(1024, 256, 1, False),
            Bottleneck(1024, 256, 1, False),
            Bottleneck(1024, 256, 1, False),
            Bottleneck(1024, 256, 1, False),
            Bottleneck(1024, 256, 1, False)
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, 2, True),
            Bottleneck(2048, 512, 1, False),
            Bottleneck(2048, 512, 1, False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 线性网络
        self.stage1 = nn.Sequential(
            nn.Linear(in_features=16, out_features=256, bias=True),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.stage2 = nn.Sequential(
            nn.Linear(256, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.stage3 = nn.Sequential(
            nn.Linear(512, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.stage4 = nn.Sequential(
            nn.Linear(1024, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.CTS = CTS(2048, 8)

        self.RSCA = RSCA()
        self.fc = nn.Sequential(
            nn.Linear(in_features=7680, out_features=3840, bias=True),
            nn.BatchNorm1d(3840),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=3840, out_features=512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=2, bias=True)
        )

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        y = self.stage1(y)
        a1 = x
        b1 = y
        c1 = self.RSCA(a1, b1)

        x = self.layer2(x)
        y = self.stage2(y)
        a2 = x
        b2 = y
        c2 = self.RSCA(a2, b2)

        x = self.layer3(x)
        y = self.stage3(y)
        a3 = x
        b3 = y
        c3 = self.RSCA(a3, b3)

        x = self.layer4(x)
        x = self.CTS(x)
        y = self.stage4(y)
        a4 = x
        b4 = y
        c4 = self.RSCA(a4, b4)

        cat_feature = torch.cat((c1, c2, c3, c4), dim=1)
        cat_feature = self.fc(cat_feature)

        return cat_feature
