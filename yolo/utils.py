import torch
from torch import nn
from typing import List

def make_conv_bn_layer(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, with_pool: bool = True):
    """
    Make a convolutional layer with batch normalization and leaky ReLU activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolutional operation.
        padding: Padding of the convolutional operation.
        with_pool: Whether to add a max pooling layer.
    """
    layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
    ]
    if with_pool:
        layers.append(nn.MaxPool2d(2, 2))

    return nn.Sequential(*layers)


