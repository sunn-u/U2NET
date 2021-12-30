import copy
import torch
import torch.nn as nn

from sunn_models.models.base import conv_block, conv_sig_block, down_sampling_layer, up_sampling_layer


# todo : 채널 input - output 사이즈와 feature map 자체의 사이즈 헷갈리지 않기


class ResidualUBlocks(nn.Module):
    def __init__(self, L, input_f, interval_f, output_f, dilation, padding):
        super(ResidualUBlocks, self).__init__()

        self.init_layer = self._make_init_layer(input_f, output_f)
        self.encode_blocks = self._make_encode_blocks(L, interval_f, output_f)
        self.dilation_layer = self._make_dilation_layer(interval_f, dilation, padding)
        self.decode_blocks = self._make_decode_blocks(L, interval_f, output_f)

    def _make_init_layer(self, input_f, output_f):
        return conv_block(input_f, output_f)

    def _make_dilation_layer(self, interval_f, dilation, padding):
        return conv_block(interval_f, interval_f, dilation=dilation, padding=padding)

    def _make_encode_blocks(self, L, interval_f, output_f):
        encode_blocks = []
        encode_blocks.append(conv_block(output_f, interval_f))
        for idx in range(L - 2):
            encode_blocks.append(conv_block(interval_f, interval_f))

        # todo : nn.Sequential
        return nn.Sequential(*encode_blocks)

    def _make_decode_blocks(self, L, interval_f, output_f):
        decode_blocks = []
        for idx in range(L - 2):
            decode_blocks.append(conv_block(2 * interval_f, interval_f))
        decode_blocks.append(conv_block(2 * interval_f, output_f))

        # todo : nn.Sequential
        return nn.Sequential(*decode_blocks)

    def forward(self, x):
        # 1. an input convolution layer
        init = self.init_layer(x)
        out = copy.deepcopy(init.data)

        # 2. a U-Net like symmetric encoder-decoder structure with height of L
        encode_outputs = []
        for encode_block in self.encode_blocks:
            out = down_sampling_layer(out)
            out = encode_block(out)
            encode_outputs.append(out)

        out = self.dilation_layer(out)

        # todo : upsampling과 cat의 위치
        for encode_output, decode_block in zip(encode_outputs[::-1], self.decode_blocks):
            out = torch.cat((encode_output, out), dim=1)
            out = up_sampling_layer(out)
            out = decode_block(out)

        # 3. a residual connection
        return out + init


class Residual4FUBlocks(nn.Module):
    def __init__(self, input_f, interval_f, output_f, dilation, padding):
        super(Residual4FUBlocks, self).__init__()
        self.init_layer = self._make_init_layer(input_f, output_f)
        self.left_layers = self._make_encode_blocks(interval_f, output_f, dilation, padding)
        self.center_layer = self._make_dilation_layer(interval_f, dilation=dilation**3, padding=padding**3)
        self.right_layers = self._make_decode_blocks(interval_f, output_f, dilation, padding)

    def _make_init_layer(self, input_f, output_f):
        return conv_block(input_f, output_f)

    def _make_dilation_layer(self, interval_f, dilation, padding):
        return conv_block(interval_f, interval_f, dilation=dilation, padding=padding)

    def _make_encode_blocks(self, interval_f, output_f, dilation, padding):
        encode_blocks = []
        encode_blocks.append(conv_block(output_f, interval_f))
        encode_blocks.append(conv_block(interval_f, interval_f, dilation=dilation**1, padding=padding**1))
        encode_blocks.append(conv_block(interval_f, interval_f, dilation=dilation**2, padding=padding**2))

        return nn.Sequential(*encode_blocks)

    def _make_decode_blocks(self, interval_f, output_f, dilation, padding):
        decode_blocks = []
        decode_blocks.append(conv_block(2*interval_f, interval_f, dilation=dilation**2, padding=padding**2))
        decode_blocks.append(conv_block(2*interval_f, interval_f, dilation=dilation**1, padding=padding**1))
        decode_blocks.append(conv_block(2*interval_f, output_f))

        return nn.Sequential(*decode_blocks)

    def forward(self, x):
        # 1. an input convolution layer
        init = self.init_layer(x)
        out = copy.deepcopy(init.data)

        # 2. a U-Net like symmetric encoder-decoder structure with multi-dilation
        left_outputs = []
        for left_block in self.left_layers:
            out = left_block(out)
            left_outputs.append(out)

        out = self.center_layer(out)

        # todo : 너무 지저분...
        for left_output, right_block in zip(left_outputs[::-1], self.right_layers):
            out = torch.cat((left_output, out), dim=1)
            out = right_block(out)

        # 3. a residual connection
        out += init

        return out


class U2NET(nn.Module):
    def __init__(self, rsu_dict: dict):
        super(U2NET, self).__init__()
        # todo : 차후 config 로 수정
        self.rsu_dict = rsu_dict

        self.encoder_rsu, self.f_rsu, self.decoder_rsu = self._stack_rsu()
        self.encoder_rsu = self.encoder_rsu
        self.f_rsu = self.f_rsu
        self.decoder_rsu = self.decoder_rsu

    def _stack_rsu(self):
        encoder_rsu = []
        f_rsu = []
        decoder_rsu = []

        for block_name, (input_channels, interval_channels, output_channels) in self.rsu_dict.items():
            if 'F' in block_name:
                f_rsu.append(Residual4FUBlocks(input_f=input_channels,
                                               interval_f=input_channels,
                                               output_f=output_channels,
                                               dilation=2,
                                               padding=2))
            else:
                L = int(block_name[-1])
                if 'E' in block_name:
                    encoder_rsu.append(ResidualUBlocks(L=L,
                                                       input_f=input_channels,
                                                       interval_f=interval_channels,
                                                       output_f=output_channels,
                                                       dilation=2,
                                                       padding=2))
                else:
                    decoder_rsu.append(ResidualUBlocks(L=L,
                                                       input_f=input_channels,
                                                       interval_f=interval_channels,
                                                       output_f=output_channels,
                                                       dilation=2,
                                                       padding=2))

        return nn.Sequential(*encoder_rsu), nn.Sequential(*f_rsu), nn.Sequential(*decoder_rsu)

    def forward(self, x):
        num, channel, width, height = x.shape

        out = x
        en_outputs = []
        for en_block in self.encoder_rsu:
            out = en_block(out)
            en_outputs.append(out)
            out = down_sampling_layer(out)

        #####################################################

        sups = []
        out = self.f_rsu[0](out)
        f_output = copy.deepcopy(out.data)
        out = down_sampling_layer(out)
        out = self.f_rsu[1](out)
        sups.append(out)
        out = up_sampling_layer(out)
        out = torch.cat((f_output, out), dim=1)
        out = self.f_rsu[2](out)
        sups.append(out)

        #####################################################

        for en_output, de_block in zip(en_outputs[::-1], self.decoder_rsu):
            out = up_sampling_layer(out)
            out = torch.cat((en_output, out), dim=1)
            out = de_block(out)
            sups.append(out)

        #####################################################

        sps = []
        # todo : device 를 밖에서 지정해주려면 미리 conv_sig_block 을 만들어둬야 하고 그럴려면 cha 크기를.....
        # 임시....
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for sup in sups:
            _, cha, _, _ = sup.shape
            # sps_block = conv_sig_block(cha, cha).to(device)
            sps_block = conv_sig_block(cha, 1).to(device)
            sup = sps_block(sup)

            upsample = nn.Upsample(size=(width, height))
            sup = upsample(sup)
            sps.append(sup)

        fuse_masks = torch.cat(sps, dim=1)
        _, cha, _, _ = fuse_masks.shape
        # sps_block = conv_sig_block(cha, channel, kernel_size=1).to(device)
        sps_block = conv_sig_block(cha, 1, kernel_size=1).to(device)
        fuse_mask = sps_block(fuse_masks)

        return sps, fuse_mask
