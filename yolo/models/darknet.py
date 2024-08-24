import torch
from torch import nn

class DarkNet(nn.Module):
    def __init__(self, in_channels: int = 3):
        self.__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=1)
        # TODO
        raise NotImplementedError("DarkNet constructor not implemented")

    def forward(self, x) -> torch.Tensor:
        # TODO
        raise NotImplementedError("DarkNet forward not implemented")

    def _make_conv_bn_layer(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, with_pool: bool = True):
        layers = [
                nn.Conv2d(in_channels, out_channnels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
        ]
        if with_pool:
            layers.append(nn.MaxPool2d(2, 2))

        return nn.Sequential(*layers)

