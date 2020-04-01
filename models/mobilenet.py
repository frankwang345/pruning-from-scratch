import torch.nn as nn
import math


def expanded_cfg(c):
    v1_cfg = [c, 2*c, 4*c, 4*c, 8*c, 8*c, 16*c, 16*c, 16*c, 16*c, 16*c, 16*c, 32*c, 32*c]
    v2_cfg = [32, 96, 144, 144, 192, 192, 192, 384, 384, 384, 384, 576, 576, 576, 960, 960, 960, 1280]
    multiplier = c / 32
    if c != 32:
        v2_cfg = [int(v * multiplier) for v in v2_cfg]

    cfg = {
        'mobilenet_v1': v1_cfg,
        'mobilenet_v2': v2_cfg
    }
    return cfg


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvDepthWise(nn.Module):

    def __init__(self, inp, oup, stride):
        super(ConvDepthWise, self).__init__()
        self.conv1 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x



class MobileNet(nn.Module):
    def __init__(self, n_class, in_channel=32, multiplier=1.0, cfg=None):
        super(MobileNet, self).__init__()
        # original
        if cfg is None:
            cfg = expanded_cfg(in_channel)['mobilenet_v1']

        self.conv1 = ConvBNReLU(3, cfg[0], 3, 2, 1)
        self.features = self._make_layers(cfg[0], cfg[1:], ConvDepthWise)
        self.pool = nn.AvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.pool(x)  # global average pooling
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

    def _make_layers(self, in_planes, cfg, layer):
        layers = []
        for i, x in enumerate(cfg):
            out_planes = x
            stride = 2 if i in [1, 3, 5, 11] else 1
            layers.append(layer(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v1(num_class, in_channel=32, multiplier=1.0, cfg=None):
    return MobileNet(num_class, in_channel, multiplier, cfg)


class InvertedBlock(nn.Module):
    def __init__(self, inp, oup, hid, stride):
        super(InvertedBlock, self).__init__()
        self.hid = hid
        # pw
        self.conv1 = nn.Conv2d(inp, hid, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hid)
        self.relu1 = nn.ReLU6(inplace=True)
        # dw
        self.conv2 = nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False)
        self.bn2 = nn.BatchNorm2d(hid)
        self.relu2 = nn.ReLU6(inplace=True)
        # pw-linear
        self.conv3 = nn.Conv2d(hid, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, hid, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        if hid == inp:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False),
                nn.BatchNorm2d(hid),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hid, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = InvertedBlock(inp, oup, hid, stride)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, in_channel=32, multiplier=1.0, cfg=None):
        super(MobileNetV2, self).__init__()
        output_channels = [
            16, 24, 24, 32, 32, 32, 64, 64, 64, 64,
            96, 96, 96, 160, 160, 160, 320
        ]
        for i in range(len(output_channels)):
            output_channels[i] = int(multiplier * output_channels[i])

        if cfg is None:
            cfg = expanded_cfg(in_channel)['mobilenet_v2']

        self.features = [ConvBNReLU(3, cfg[0], kernel_size=3, stride=2, padding=1)]
        # building inverted residual blocks
        inp = cfg[0]
        for j, (hid, oup) in enumerate(zip(cfg[:-1], output_channels)):
            if j in [1, 3, 6, 13]:
                stride = 2
            else:
                stride = 1
            self.features.append(InvertedResidual(inp, oup, hid, stride))
            inp = oup

        # building last several layers
        self.features.append(ConvBNReLU(inp, cfg[-1], kernel_size=1, stride=1, padding=0))
        self.features.append(nn.AvgPool2d(7))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenet_v2(num_class, in_channel=32, multiplier=1.0, cfg=None):
    return MobileNetV2(num_class, in_channel, multiplier, cfg)
