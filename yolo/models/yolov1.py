import torch
from torch import nn
from darknet import DarkNet

# TODO

class YOLOv1(nn.Module):
    """
    YOLOv1 object detection neural network.
    First version of the family. 
    """
    def __init__(self, in_channels: int = 3) -> None:
        # TODO
        self.in_channels = in_channels
        self.backbone = DarkNet(in_channels)
        raise NotImplementedError("YOLOv1 constructor not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError("YOLOv1 forward not implemented")

