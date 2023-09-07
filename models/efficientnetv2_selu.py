# https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021).
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""

import torch
import torch.nn as nn
import math
from .layers import Output

__all__ = ['efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l', 'efficientnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            torch.nn.SELU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        torch.nn.SELU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        torch.nn.SELU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                torch.nn.SELU(),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                torch.nn.SELU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                torch.nn.SELU(),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv3d(inp, hidden_dim, 3, stride, 1, bias=False),
                torch.nn.SELU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                torch.nn.SELU(),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, n_features, channels=3, num_classes=2, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs
        self.n_features = n_features

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(channels, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        # output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = Output(output_channel + self.n_features, num_classes)
        # self.classifier.__class__.__name__ = 'Output'

        self._initialize_weights()

    def forward(self, x, features):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Add features
        if self.n_features > 0:
            x = torch.cat([x, features], dim=1)

        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        weight_init_name = 'kaiming_normal'
        kaiming_a = None
        kaiming_mode = 'fan_in'
        kaiming_nonlinearity = 'linear'

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=kaiming_a, mode=kaiming_mode,
                                              nonlinearity=kaiming_nonlinearity)
                if m.bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)  # if fan_in > 0 else 0
                    torch.nn.init.uniform_(m.bias, -bound, bound)


def efficientnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def efficientnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def efficientnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def efficientnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def get_efficientnetv2_selu(model_name, channels, n_features, num_classes):

    if model_name == 'efficientnetv2_s_selu':
        model = efficientnetv2_s(channels=channels, n_features=n_features, num_classes=num_classes)
    elif model_name == 'efficientnetv2_m_selu':
        model = efficientnetv2_m(channels=channels, n_features=n_features, num_classes=num_classes)
    elif model_name == 'efficientnetv2_l_selu':
        model = efficientnetv2_l(channels=channels, n_features=n_features, num_classes=num_classes)
    elif model_name == 'efficientnetv2_xl_selu':
        model = efficientnetv2_xl(channels=channels, n_features=n_features, num_classes=num_classes)
    else:
        raise ValueError('Model_name = {} is not valid!'.format(model_name))

    return model

