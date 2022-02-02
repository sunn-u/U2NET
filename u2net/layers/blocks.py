# Coding by SunWoo(tjsntjsn20@gmail.com)

import copy
import torch
import torch.nn as nn
from fvcore.common.registry import Registry

from . import convolution_layer
from u2net.data.types import Tensor
from u2net.layers.encoder import Encoder, Encoder4F
from u2net.layers.decoder import Decoder, Decoder4F

BLOCK_REGISTRY = Registry("BLOCK")
BLOCK_REGISTRY.__doc__ = " Registry for Blocks "

@BLOCK_REGISTRY.register()
class RSU(nn.Module):
    '''
        Residual U Blocks
    '''

    def __init__(
        self,
        L: int,
        input_channels: int,
        middle_channels: int,
        output_channels: int,
        kernel_size=3,
        dilation=1,
        padding=1
    ):
        super().__init__()

        self.init_layer = convolution_layer(in_c=input_channels, out_c=output_channels)
        self.dilation_layer = convolution_layer(in_c=middle_channels, out_c=middle_channels, dilation=2, padding=2)
        self.encoders = Encoder(L=L, input_channels=output_channels, output_channels=middle_channels)
        self.decoders = Decoder(L=L, input_channels=middle_channels*2, output_channels=output_channels)

    def forward(self, x: Tensor) -> Tensor:
        # 1. an input convolution layer
        init = self.init_layer(x)

        # 2. a U-Net like symmetric encoder-decoder structure with height of L
        encoder_features = []
        out = copy.deepcopy(init.data)
        for idx in range(len(self.encoders.block)):
            out = self.encoders.block[idx](out)
            encoder_features.insert(0, out)

        out = self.dilation_layer(out)

        for idx in range(len(self.decoders.block)):
            out = torch.cat((encoder_features[idx], out), dim=1)
            out = self.decoders.block[idx](out)

        # 3. a residual connection
        return init + out


@BLOCK_REGISTRY.register()
class RSU4F(nn.Module):
    '''
        Residual U 4F Blocks
    '''

    def __init__(
        self,
        L: int,
        input_channels: int,
        middle_channels: int,
        output_channels: int,
        kernel_size=3
    ):
        super().__init__()

        self.init_layer = convolution_layer(in_c=input_channels, out_c=output_channels)
        self.dilation_layer = convolution_layer(in_c=middle_channels, out_c=middle_channels, dilation=2**(L-1), padding=2**(L-1))
        self.encoders = Encoder4F(L=L, input_channels=output_channels, output_channels=middle_channels)
        self.decoders = Decoder4F(L=L, input_channels=middle_channels*2, output_channels=output_channels)

    def forward(self, x: Tensor) -> Tensor:
        # 1. an input convolution layer
        init = self.init_layer(x)

        # 2. a U-Net like symmetric encoder-decoder structure with height of L
        encoder_features = []
        out = copy.deepcopy(init.data)
        for idx in range(len(self.encoders.block)):
            out = self.encoders.block[idx](out)
            encoder_features.insert(0, out)

        out = self.dilation_layer(out)

        for idx in range(len(self.decoders.block)):
            out = torch.cat((encoder_features[idx], out), dim=1)
            out = self.decoders.block[idx](out)

        # 3. a residual connection
        return init + out
