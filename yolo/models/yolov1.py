import torch
from torch import nn
from models.darknet import DarkNet
from typing import Any, List
from utils.yolo import make_conv_bn_layer

# TODO

class YOLOv1(nn.Module):
    """
    YOLOv1 object detection neural network.
    First version of the family. 
    """
    def __init__(self, config: Any) -> None:
        """
        YOLOv1 class constructor.

        Args:
        - config: Any: configuration object as specified in the configs/yolov1.yaml file.
        """
        super().__init__()
        self.grid_size = int(config['grid_size'])
        self.num_classes = config['num_classes']

        # initialize the backbone from the configuration
        self.backbone = DarkNet(config['backbone'])

        # initialize the classifier layers from the configuration
        self.classifier = self.generate_classifier_layers(config)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        YOLOv1 forward pass.

        Args:
        - x: torch.Tensor: input image(s) of shape (B, C, H, W).

        Returns:
        - torch.Tensor: output tensor of shape (B, S, S, 5 * B + C), where S is the grid size.
        """

        features = self.backbone(x)
        bboxes = self.classifier(features)

        return bboxes
    
    def generate_classifier_layers(self, config: Any) -> nn.Module:
        """
        Generates the classifier layers of the YOLOv1 network from the configuration.

        Args:
        - config: Any: classifier object.

        Returns:
        - nn.Sequential: classifier layers as a sequential module.
        """

        # create a list of blocks
        blocks: List[nn.Module] = []

        # iterate the conv blocks description
        for block in config['classifier']['conv_blocks']:

            # create a list of layers for the block
            layers: List[nn.Module] = []

            in_channels = 1024

            # iterate the layers
            for layer in block['layers']:
                blocks.append(make_conv_bn_layer(in_channels, layer[0], layer[1], layer[2], layer[3], False))
                in_channels = layer[0]

        # create a sequential module from the blocks
        return nn.Sequential(*blocks)
