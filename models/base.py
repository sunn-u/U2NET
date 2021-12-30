import torch.nn as nn
import torch.nn.functional as F


def weight_init_xavier(submodule):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1

    if isinstance(submodule, nn.Conv2d):
        nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)

def conv_block(input_f, output_f, kernel_size=3, dilation=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=input_f,
            out_channels=output_f,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        ),
        nn.BatchNorm2d(num_features=output_f),
        nn.ReLU()
    )

def conv_sig_block(input_f, output_f, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=input_f,
            out_channels=output_f,
            kernel_size=kernel_size
        ),
        nn.Sigmoid()
    )

def down_sampling_layer(x):
    return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)

def up_sampling_layer(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
