import math
import torch.nn as nn


def expanded_cfg(c):
    defaultcfg = {
        'vgg11_bn' : [c, 'M', c*2, 'M', c*4, c*4, 'M', c*8, c*8, 'M', c*8, c*8],
        'vgg13_bn' : [c, c, 'M', c*2, c*2, 'M', c*4, c*4, 'M', c*8, c*8, 'M', c*8, c*8],
        'vgg16_bn' : [c, c, 'M', c*2, c*2, 'M', c*4, c*4, c*4, 'M', c*8, c*8, c*8, 'M', c*8, c*8, c*8],
        'vgg19_bn' : [c, c, 'M', c*2, c*2, 'M', c*4, c*4, c*4, c*4, 'M', c*8, c*8, c*8, c*8, 'M', c*8, c*8, c*8, c*8],
    }
    return defaultcfg


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CifarVGG(nn.Module):
    def __init__(self, depth, expanded_inchannel, num_classes=10, cfg=None):
        super(CifarVGG, self).__init__()
        if cfg is None:
            defaultcfg = expanded_cfg(expanded_inchannel)
            cfg = defaultcfg['vgg' + str(depth) + '_bn']

        self.feature = self.make_layers(cfg)
        self.num_classes = num_classes
        self.classifier = nn.Linear(cfg[-1], num_classes)
        self._initialize_weights()

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [ConvBNReLU(in_channels, v, kernel_size=3, padding=1, bias=False)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg16_bn(num_classes, expanded_inchannel=64, cfg=None):
    return CifarVGG(16, expanded_inchannel, num_classes, cfg)

def vgg19_bn(num_classes, expanded_inchannel=64, cfg=None):
    return CifarVGG(19, expanded_inchannel, num_classes, cfg)
