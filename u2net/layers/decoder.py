# Coding by SunWoo(tjsntjsn20@gmail.com)

import torch.nn as nn

from . import  convolution_layer, up_convolution_layer
from u2net.data.types import Tensor


class Decoder(nn.Module):
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

        # Build decoding-block.
        self.block = nn.Sequential(*self.build_block())

    def build_block(self):
        decoders = []
        for idx in range(self.L - 2):
            decoders.append(
                up_convolution_layer(
                    in_c=self.input_channels,
                    out_c=int(self.input_channels/2)
                ))
        decoders.append(convolution_layer(in_c=self.input_channels, out_c=self.output_channels))
        return decoders

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Decoder4F(nn.Module):
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
        decoders = []
        for idx in range(self.L-1, 1, -1):
            decoders.append(
                convolution_layer(
                    in_c=self.input_channels,
                    out_c=int(self.input_channels/2),
                    dilation=2**idx,
                    padding=2**idx
                ))
        decoders.append(convolution_layer(in_c=self.input_channels, out_c=self.output_channels))
        return decoders

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
