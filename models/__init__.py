from .vgg import vgg16_bn, vgg19_bn
from .resnet import resnet20, resnet32, \
    resnet44, resnet56, resnet110, resnet50
from .mobilenet import mobilenet_v1, mobilenet_v2

def expanded_cfg(c):
    v1_cfg = [c, 2*c, 4*c, 4*c, 8*c, 8*c, 16*c, 16*c, 16*c, 16*c, 16*c, 16*c, 32*c, 32*c]
    v2_cfg = [32, 96, 144, 144, 192, 192, 192, 384, 384, 384, 384, 576, 576, 576, 960, 960, 960, 1280]
    multiplier = c / 32
    if c != 32:
        v2_cfg = [int(v * multiplier) for v in v2_cfg]

    defaultcfg = {
        'vgg11_bn' : [c, 'M', c*2, 'M', c*4, c*4, 'M', c*8, c*8, 'M', c*8, c*8],
        'vgg13_bn' : [c, c, 'M', c*2, c*2, 'M', c*4, c*4, 'M', c*8, c*8, 'M', c*8, c*8],
        'vgg16_bn' : [c, c, 'M', c*2, c*2, 'M', c*4, c*4, c*4, 'M', c*8, c*8, c*8, 'M', c*8, c*8, c*8],
        'vgg19_bn' : [c, c, 'M', c*2, c*2, 'M', c*4, c*4, c*4, c*4, 'M', c*8, c*8, c*8, c*8, 'M', c*8, c*8, c*8, c*8],
        'resnet20': [c] * 3 + [c*2] * 3 + [c*4] * 3,
        'resnet32': [c] * 5 + [c*2] * 5 + [c*4] * 5,
        'resnet44': [c] * 7 + [c*2] * 7 + [c*4] * 7,
        'resnet56': [c] * 9 + [c*2] * 9 + [c*4] * 9,
        'resnet110': [c] * 18 + [c*2] * 18 + [c*4] * 18,
        'mobilenet_v1': v1_cfg,
        'mobilenet_v2': v2_cfg,
        'resnet50': [c, c, c, c, c, c, 2*c, 2*c, 2*c, 2*c, 2*c, 2*c, 2*c, 2*c,
                   4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c, 4*c,
                   8*c, 8*c, 8*c, 8*c, 8*c, 8*c]
    }
    return defaultcfg
