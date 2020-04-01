import math
import torch.nn as nn


def expanded_cfg(c):
    defaultcfg = {
        'resnet20': [c] * 3 + [c*2] * 3 + [c*4] * 3,
        'resnet32': [c] * 5 + [c*2] * 5 + [c*4] * 5,
        'resnet44': [c] * 7 + [c*2] * 7 + [c*4] * 7,
        'resnet56': [c] * 9 + [c*2] * 9 + [c*4] * 9,
        'resnet110': [c] * 18 + [c*2] * 18 + [c*4] * 18,
        'resnet50': [c, c, c, c, c, c, 2*c, 2*c, 2*c, 2*c, 2*c, 2*c, 2*c, 2*c,
                     4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c,
                     8*c, 8*c, 8*c, 8*c, 8*c, 8*c]
    }
    return defaultcfg

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, mid_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class CifarResNet(nn.Module):
    def __init__(self, depth, expanded_inchannel, num_classes=10, cfg=None):
        super(CifarResNet, self).__init__()
        if cfg is None:
            defaultcfg = expanded_cfg(expanded_inchannel)
            cfg = defaultcfg['resnet' + str(depth)]
        n_blocks = (depth - 2) // 6

        self.in_planes = expanded_inchannel
        self.conv1 = nn.Conv2d(3, expanded_inchannel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_inchannel)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, expanded_inchannel, cfg[0:n_blocks], stride=1)
        self.layer2 = self._make_layer(BasicBlock, expanded_inchannel*2, cfg[n_blocks:2*n_blocks], stride=2)
        self.layer3 = self._make_layer(BasicBlock, expanded_inchannel*4, cfg[2*n_blocks:], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(expanded_inchannel*4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channel, cfg, stride):
        layers = []
        for i in range(0, len(cfg)):
            layers.append(block(self.in_planes, cfg[i], channel, stride))
            stride = 1
            self.in_planes = channel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes, expanded_inchannel=16, cfg=None):
    return CifarResNet(20, expanded_inchannel, num_classes, cfg)

def resnet32(num_classes, expanded_inchannel=16, cfg=None):
    return CifarResNet(32, expanded_inchannel, num_classes, cfg)

def resnet44(num_classes, expanded_inchannel=16, cfg=None):
    return CifarResNet(44, expanded_inchannel, num_classes, cfg)

def resnet56(num_classes, expanded_inchannel=16, cfg=None):
    return CifarResNet(56, expanded_inchannel, num_classes, cfg)

def resnet110(num_classes, expanded_inchannel=16, cfg=None):
    return CifarResNet(110, expanded_inchannel, num_classes, cfg)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, mid_planes1, mid_planes2, out_planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes1, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_planes1, mid_planes2, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_planes2, out_planes, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes, expanded_inchannel=64, multiplier=1.0, cfg=None):
        super(ResNet50, self).__init__()
        if cfg is None:
            c = expanded_inchannel
            cfg = [c, c, c, c, c, c, 2*c, 2*c, 2*c, 2*c, 2*c, 2*c, 2*c, 2*c,
                   4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c,
                   8*c, 8*c, 8*c, 8*c, 8*c, 8*c]

        output_channels = [64, 256, 256, 256, 512, 512, 512, 512,
                           1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048]

        for i in range(len(output_channels)):
            output_channels[i] = int(multiplier * output_channels[i])

        self.in_planes = output_channels[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(cfg[:6], output_channels[1:4], stride=1)
        self.layer2 = self._make_layer(cfg[6:14], output_channels[4:8], stride=2)
        self.layer3 = self._make_layer(cfg[14:26], output_channels[8:14], stride=2)
        self.layer4 = self._make_layer(cfg[26:], output_channels[14:], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(output_channels[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, cfg, output_channels, stride):
        layers = [Bottleneck(self.in_planes, cfg[0], cfg[1], output_channels[0], stride)]
        self.in_planes = output_channels[0]
        for i in range(1, len(output_channels)):
            layers.append(Bottleneck(self.in_planes, cfg[2*i], cfg[2*i+1], output_channels[i]))
            self.in_planes = output_channels[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(num_classes, in_channel=64, multiplier=1.0, cfg=None):
    return ResNet50(num_classes, in_channel, multiplier, cfg)