# Coding by SunWoo(tjsntjsn20@gmail.com)

import torch.nn as nn

from . import  convolution_layer, down_convolution_layer
from u2net.data.types import Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        L: int,
        input_channels: int,
        output_channels: int,
        kernel_size=3,
        dilation=1,
        padding=1
    ):
        super().__init__()

        self.L = L
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Build encoding-block.
        self.block = nn.Sequential(*self.build_block())

    def build_block(self):
        encoders = []
        encoders.append(convolution_layer(in_c=self.input_channels, out_c=self.output_channels))
        for idx in range(self.L-2):
            encoders.append(
                down_convolution_layer(
                    in_c=self.output_channels,
                    out_c=self.output_channels
                ))
        return encoders

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Encoder4F(nn.Module):
    def __init__(
        self,
        L: int,
        input_channels: int,
        output_channels: int,
        kernel_size=3
    ):
        super().__init__()

        self.L = L
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Build encoding-block.
        self.block = nn.Sequential(*self.build_block())

    def build_block(self):
        encoders = []
        encoders.append(convolution_layer(in_c=self.input_channels, out_c=self.output_channels))
        for idx in range(1, self.L-1):
            encoders.append(
                convolution_layer(
                    in_c=self.output_channels,
                    out_c=self.output_channels,
                    dilation=2**idx,
                    padding=2**idx,
                ))
        return encoders

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
