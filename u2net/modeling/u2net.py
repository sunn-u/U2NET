# Coding by SunWoo(tjsntjsn20@gmail.com)

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.common.registry import Registry

from u2net.data.types import Tensor
from u2net.layers.blocks import BLOCK_REGISTRY
from u2net.data.types import DictConfigs
from u2net.layers.layer import convolution_layer_without_bn, sigmoid_convolution_layer

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = " Registry for Model "


@MODEL_REGISTRY.register()
class U2NET(nn.Module):
    def __init__(self, configs: DictConfigs):
        super(U2NET, self).__init__()
        rsu_dict = configs.model.meta_architecture.architecture_dict
        img_size = configs.data.dataloader.transforms.Resize.size

        encoder_blocks, decoder_blocks, side_blocks = self.build_blocks(rsu_dict, img_size)
        self.encoder_blocks = nn.Sequential(*encoder_blocks)
        self.decoder_blocks = nn.Sequential(*decoder_blocks)
        self.side_blocks = nn.Sequential(*side_blocks)

    def build_blocks(self, rsu_dict: dict, size: int):
        # Build encoder-blocks and decoder-blocks.
        encoder_blocks, decoder_blocks, side_blocks = [], [], []
        for block_info, parameters in rsu_dict.items():
            block_name = block_info.split('-')[0]
            block_part = block_info.split('-')[-1][0]
            L, input_channels, middel_channels, output_channels = parameters

            # build rsu or rsu-4f.
            block = BLOCK_REGISTRY.get(block_name.upper())(L, input_channels, middel_channels, output_channels)
            if block_part == 'e':
                encoder_blocks.append(block)
            else:
                decoder_blocks.append(block)
                side_blocks.append(sigmoid_convolution_layer(size=(size, size), in_c=output_channels, out_c=1))
        side_blocks.insert(0, copy.deepcopy(side_blocks[0]))
        side_blocks.append(convolution_layer_without_bn(in_c=len(side_blocks), out_c=1, kernel_size=1, padding=0, sigmoid=True))

        return encoder_blocks, decoder_blocks, side_blocks

    def forward(self,x: Tensor) -> Tensor:
        encoder_features = []
        sups = []
        for idx in range(len(self.encoder_blocks)-1):
            x = self.encoder_blocks[idx](x)
            batch, channel, height, width = x.shape
            x = F.interpolate(x, size=(int(height/2), int(width/2)), mode='bilinear', align_corners=True)
            encoder_features.insert(0, x)
        x = self.encoder_blocks[-1](x)
        sups.append(x)

        for idx in range(len(self.decoder_blocks)):
            x = torch.concat((encoder_features[idx], x), dim=1)
            batch, channel, height, width = x.shape
            x = F.interpolate(x, size=(height*2, width*2), mode='bilinear', align_corners=True)
            x = self.decoder_blocks[idx](x)
            sups.append(x)

        # calculate Sups.
        masks = []
        for idx, sup in enumerate(sups):
            mask = self.side_blocks[idx](sup)
            masks.append(mask)
        fuse_mask = self.side_blocks[-1](torch.cat((masks), dim=1))

        return masks, fuse_mask
