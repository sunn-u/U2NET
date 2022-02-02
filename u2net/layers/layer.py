# Coding by SunWoo(tjsntjsn20@gmail.com)

import torch.nn as nn


def weight_init_xavier(submodule):
    if isinstance(submodule, nn.Conv2d):
        nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)


def convolution_layer(in_c: int, out_c: int, kernel_size=3, dilation=1, padding=1, sigmoid=False):
    activation_layer = nn.Sigmoid() if sigmoid else nn.ReLU()
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        ),
        nn.BatchNorm2d(num_features=out_c),
        activation_layer
    )


def convolution_layer_without_bn(in_c: int, out_c: int, kernel_size=3, dilation=1, padding=1, sigmoid=False):
    activation_layer = nn.Sigmoid() if sigmoid else nn.ReLU()
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        ),
        activation_layer
    )


def down_convolution_layer(in_c: int, out_c: int, kernel_size=3, dilation=1, padding=1):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        ),
        nn.BatchNorm2d(num_features=out_c),
        nn.ReLU()
    )


def up_convolution_layer(in_c: int, out_c: int, kernel_size=3, dilation=1, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=2, stride=2),
        nn.Conv2d(
            in_channels=out_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        ),
        nn.BatchNorm2d(num_features=out_c),
        nn.ReLU()
    )


def sigmoid_convolution_layer(size: tuple, in_c: int, out_c: int, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            padding=1
        ),
        nn.Upsample(size=size, mode='bilinear', align_corners=True),
        nn.Sigmoid()
    )
