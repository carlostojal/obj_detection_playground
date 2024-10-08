import torch
from torch import nn
from typing import List, Any
from yolo.utils import make_conv_bn_layer

class DarkNet(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.in_channels = config['in_channels']
        self.conv_blocks = self.generate_conv_blocks(config)

    def forward(self, x) -> torch.Tensor:
        x = self.conv_blocks(x)
        return x
    
    def generate_conv_blocks(self, config: Any) -> nn.Sequential:

        in_channels = self.in_channels

        blocks: List[nn.Module] = []

        # iterate the conv blocks description
        for block in config['conv_blocks']:
            # iterate the block layers
            n_layers = len(block['layers'])
            cur_layer = 0
            for layer in block['layers']:
                blocks.append(make_conv_bn_layer(in_channels, layer[0], layer[1], layer[2], layer[3], cur_layer == n_layers-1))
                in_channels = layer[0]
                cur_layer += 1

        return nn.Sequential(*blocks)
