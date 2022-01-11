# Coding by SunWoo(tjsntjsn20@gmail.com)

import torch.nn as nn
import torch.nn.functional as F

from u2net.data.types import Tensor


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        scaling = kwargs.pop("scaling", None)
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.scaling = scaling
        self.norm = norm
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        if self.scaling is not None:
            x = self.scaling(x)

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
