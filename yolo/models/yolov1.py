import torch
from torch import nn
from yolo.models.darknet import DarkNet
from typing import Any, List
from yolo.utils import make_conv_bn_layer

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
        self.num_boxes = int(config['n_predictors'])
        self.num_classes = int(config['n_classes'])

        # initialize the backbone from the configuration
        self.backbone = DarkNet(config['backbone'])

        # initialize the classifier layers from the configuration
        self.classifier = self.generate_classifier_layers(config)

        # initialize the fully connected layers
        self.fc = self.generate_fc_layers(config)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        YOLOv1 forward pass.

        Args:
        - x: torch.Tensor: input image(s) of shape (B, C, H, W).

        Returns:
        - torch.Tensor: output tensor of shape (B, S, S, 5 * B + C), where S is the grid size.
        """

        # extract features using the backbone
        x = self.backbone(x)
        # pass the features through the classifier
        x = self.classifier(x)
        # flatten the tensor
        x = x.view(x.size(0), -1)
        # pass through the fully connected layers to get the bounding boxes
        bboxes = self.fc(x)
        # reshape the bounding boxes
        bboxes = bboxes.view(-1, self.grid_size, self.grid_size, (5 * self.num_boxes) + self.num_classes)

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
    
    def generate_fc_layers(self, config: Any) -> nn.Module:

        layers: List[nn.Module] = []

        # get the number of output channels from the classifier
        out_channels_classifier = int(config['classifier']['conv_blocks'][-1]['layers'][-1][0])

        # get the hidden layer dimension
        hidden_dim = int(config['classifier']['fc_layers']['dim'])

        # get the dropout rate
        dropout_rate = float(config['classifier']['fc_layers']['dropout'])

        fc1 = nn.Linear(out_channels_classifier, hidden_dim)
        layers.append(fc1)
        layers.append(nn.LeakyReLU(0.1, inplace=True))

        layers.append(nn.Dropout(dropout_rate))

        fc2 = nn.Linear(4096, self.grid_size**2 * ((5 * self.num_boxes) + self.num_classes))
        layers.append(fc2)
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)
