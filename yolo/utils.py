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

def fsoco_to_yolo_bboxes(bboxes: torch.Tensor, img_dims: torch.Tensor, grid_size: int = 7, n_predictors: int = 2, n_classes: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert bounding boxes from the FSOCO format (xyhw relative to image 0-1) to the YOLO format (xy relative to grid cell and hw relative to image dimensions 0-1).

    Args:
        bboxes (torch.Tensor): Bounding boxes in the FSOCO format shaped (batch_size, n_boxes, 5) where the last dimension is [x, y, h, w, class].
        img_dims (torch.Tensor): Image dimensions shaped (batch_size, 2) where the last dimension is [height, width].
        grid_size (int): Size of the grid in YOLO.
        n_predictors (int): Number of predictors per grid cell.
        n_classes (int): Number of classes in the dataset.

    Returns:
        (torch.Tensor) Bounding boxes in the YOLO format shaped (batch_size, S, S, n_predictors * (5 + n_classes)) where S is the grid size.
    """

    batch_size = bboxes.size(0)
    S = grid_size
    n_bboxes = bboxes.size(1)

    # calculate the grid cell size
    cell_height = img_dims[:, 0] / S
    cell_width = img_dims[:, 1] / S

    # create the grid
    grid = torch.zeros((batch_size, S, S, n_predictors * (5 + n_classes)))

    # convert the bounding box xy coordinates to center coordinates
    bboxes[:, :, 0] = bboxes[:, :, 0] + bboxes[:, :, 3] / 2
    bboxes[:, :, 1] = bboxes[:, :, 1] + bboxes[:, :, 2] / 2

    # get the grid coordinates of the bounding boxes
    bbox_x_grid = (bboxes[:, :, 0] / cell_width).floor()
    bbox_y_grid = (bboxes[:, :, 1] / cell_height).floor()

    # convert the bounding box locations to the YOLO format (center coordinate relative to grid cell and width/height)
    bboxes[:, :, 0] = (bboxes[:, :, 0] % cell_width) / cell_width
    bboxes[:, :, 1] = (bboxes[:, :, 1] % cell_height) / cell_height

    # iterate the batches
    for b in range(batch_size):
        # iterate the bounding boxes
        for i in range(n_bboxes):
            # iterate the predictors
            for j in range(n_predictors):
                # get the grid cell
                grid_cell = grid[b, int(bbox_y_grid[b, i]), int(bbox_x_grid[b, i]), j * (5 + n_classes): (j + 1) * (5 + n_classes)]
                # set the bounding box
                grid_cell[0] = bboxes[b, i, 0]
                grid_cell[1] = bboxes[b, i, 1]
                grid_cell[2] = bboxes[b, i, 2]
                grid_cell[3] = bboxes[b, i, 3]
                grid_cell[4] = 1
                grid_cell[5 + int(bboxes[b, i, 4])] = 1

    return grid
