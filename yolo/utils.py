import torch
from torch import nn
from typing import List, Tuple

def make_conv_bn_layer(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, with_pool: bool = True) -> nn.Module:
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

def fsoco_to_yolo_bboxes(bboxes: torch.Tensor, img_dims: torch.Tensor, grid_size: int, n_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert bounding boxes from the FSOCO format (xyhw relative to image 0-1) to the YOLO format (xy relative to grid cell and hw relative to image dimensions 0-1).

    Args:
        bboxes (torch.Tensor): Bounding boxes in the FSOCO format shaped (batch_size, n_boxes, 5) where the last dimension is [x, y, h, w, class].
        img_dims (torch.Tensor): Image dimensions shaped (batch_size, 2) where the last dimension is [height, width].
        grid_size (int): Size of the grid in YOLO.
        n_classes (int): Number of classes in the dataset.

    Returns:
        (torch.Tensor) Bounding boxes in the YOLO format shaped (batch_size, n_boxes, 4) where the last dimension is [x, y, h, w].
        (torch.Tensor) Class distribution for each bounding box shaped (batch_size, n_boxes, n_classes).
    """

    img_height, img_width = img_dims[:, 0], img_dims[:, 1]

    # extract coordinates and class labels
    width = bboxes[:, :, 3] * img_width
    height = bboxes[:, :, 2] * img_height
    x_center = bboxes[:, :, 0] + (width / 2)
    y_center = bboxes[:, :, 1] + (height / 2)
    class_labels = bboxes[:, :, 4]
    # create a class distribution tensor
    class_dist = torch.zeros(bboxes.shape[0], bboxes.shape[1], n_classes)
    class_dist[torch.arange(bboxes.shape[0]).unsqueeze(1).expand(-1, bboxes.shape[1]), torch.arange(bboxes.shape[1]).expand(bboxes.shape[0], -1), class_labels.long()] = 1

    # calculate grid cell indices
    cell_height, cell_width = img_height / grid_size, img_height / grid_size
    grid_x = (x_center / cell_width).long()
    grid_y = (y_center / cell_height).long()

    # calculate relative coordinates and dimensions
    x_rel = (x_center - grid_x * cell_width) / cell_width
    y_rel = (y_center - grid_y * cell_height) / cell_height
    width_rel = width / img_width
    height_rel = height / img_height

    return torch.stack([x_rel, y_rel, height_rel, width_rel], dim=-1), class_dist
