import torch
from torch import nn
from typing import List, Tuple, Any

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

def fsoco_to_yolo_bboxes(bboxes: torch.Tensor, img_dims: Tuple[int], grid_size: int = 7, n_predictors: int = 2, n_classes: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert bounding boxes from the FSOCO format (xyhw relative to image 0-1) to the YOLO format (xy relative to grid cell and hw relative to image dimensions 0-1).

    Args:
        bboxes (torch.Tensor): Bounding boxes in the FSOCO format shaped (batch_size, n_boxes, 5) where the last dimension is [x, y, h, w, class].
        img_dims (Tuple[int]): Image dimensions (height, width).
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
    cell_height, cell_width = img_dims[0] / S, img_dims[1] / S

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

def yolo_to_fsoco_bboxes(bboxes: torch.Tensor, img_dims: Tuple[int], grid_size: int = 7, n_predictors: int = 2, n_classes: int = 5) -> torch.Tensor:
    """
    Convert bounding boxes from the YOLO format (xy relative to grid cell and hw relative to image dimensions 0-1) to the FSOCO format (xyhw relative to image 0-1).

    Args:
        bboxes (torch.Tensor): Bounding boxes in the YOLO format shaped (batch_size, S, S, n_predictors * (5 + n_classes)) where S is the grid size.
        img_dims (Tuple[int]): Image dimensions (height, width).
        grid_size (int): Size of the grid in YOLO.
        n_predictors (int): Number of predictors per grid cell.
        n_classes (int): Number of classes in the dataset.

    Returns:
        (torch.Tensor) Bounding boxes in the FSOCO format shaped (batch_size, n_boxes, 5) where the last dimension is [x, y, h, w, class].
    """

    batch_size = bboxes.size(0)
    S = grid_size
    n_bboxes = S**2 * n_predictors

    # calculate the grid cell size
    cell_height, cell_width = img_dims[0] / S, img_dims[1] / S

    # create the bounding boxes tensor
    fsoco_bboxes = torch.zeros((batch_size, n_bboxes, 5))

    # iterate the batches
    for b in range(batch_size):
        # iterate the grid cells
        for gy in range(S):
            for gx in range(S):
                # iterate the predictors
                for p in range(n_predictors):
                    # get the grid cell
                    grid_cell = bboxes[b, gy, gx, p * (5 + n_classes): (p + 1) * (5 + n_classes)]
                    # get the bounding box
                    bbox = fsoco_bboxes[b, gy * S + gx * n_predictors + p]
                    # set the bounding box
                    bbox[0] = (gx + grid_cell[0]) * cell_width[b]
                    bbox[1] = (gy + grid_cell[1]) * cell_height[b]
                    bbox[2] = grid_cell[2] * cell_width[b]
                    bbox[3] = grid_cell[3] * cell_height[b]
                    bbox[4] = grid_cell[5:].argmax()

    return fsoco_bboxes

def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute the Intersection over Union (IoU) of two sets of boxes.

    Args:
        boxes1 (torch.Tensor): Boxes in the format (x, y, h, w) shaped (batch_size, n_boxes, 4).
        boxes2 (torch.Tensor): Boxes in the format (x, y, h, w) shaped (batch_size, n_boxes, 4).

    Returns:
        (torch.Tensor) IoU values shaped (batch_size, n_boxes, n_boxes).
    """

    # get the coordinates and dimensions of each box
    b1_x1, b1_y1, b1_h, b1_w = boxes1[:, :, 0], boxes1[:, :, 1], boxes1[:, :, 2], boxes1[:, :, 3]
    b2_x1, b2_y1, b2_h, b2_w = boxes2[:, :, 0], boxes2[:, :, 1], boxes2[:, :, 2], boxes2[:, :, 3]

    # convert to corner coordinates
    b1_x2, b1_y2 = b1_x1 + b1_w, b1_y1 + b1_h
    b2_x2, b2_y2 = b2_x1 + b2_w, b2_y1 + b2_h

    # get the intersection coordinates
    inter_x1 = torch.max(b1_x1.unsqueeze(2), b2_x1.unsqueeze(1))
    inter_y1 = torch.max(b1_y1.unsqueeze(2), b2_y1.unsqueeze(1))
    inter_x2 = torch.min(b1_x2.unsqueeze(2), b2_x2.unsqueeze(1))
    inter_y2 = torch.min(b1_y2.unsqueeze(2), b2_y2.unsqueeze(1))

    # calculate the intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box1_area = b1_h.unsqueeze(2) * b1_w.unsqueeze(2)
    box2_area = b2_h.unsqueeze(1) * b2_w.unsqueeze(1)

    # calculate the union area - the small epsilon is added to avoid division by zero
    union_area = box1_area + box2_area - inter_area + 1e-6

    # calculate the IoU
    iou = inter_area / union_area

    return iou
    
class YOLOv1Loss(nn.Module):
    """
    YOLOv1 loss function.
    """

    def __init__(self, conf: Any) -> None:
        """
        Initialize the YOLOv1 loss function.

        Args:
            conf (Any): Configuration object as specified in the configs/yolov1.yaml file.
        """

        super().__init__()

        self.lambda_coord = float(conf['lambda_coord']) 
        self.lambda_noobj = float(conf['lambda_noobj'])
        self.grid_size = int(conf['grid_size'])
        self.n_predictors = int(conf['n_predictors'])

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss from the ground truth and predicted bounding boxes.

        Args:
            y_true (torch.Tensor): Ground truth bounding boxes shaped (batch_size, S, S, n_predictors * (5 + n_classes)) where S is the grid size.
            y_pred (torch.Tensor): Predicted bounding boxes shaped (batch_size, S, S, n_predictors * (5 + n_classes)) where S is the grid size.

        Returns:
            (torch.Tensor) Loss value.
        """

        # get the predictions boxes and classes
        pred_boxes = y_pred[:, :, :, :self.n_predictors * 5]
        pred_classes = y_pred[:, :, :, self.n_predictors * 5:]

        # get the ground truth boxes and classes
        target_boxes = y_true[:, :, :, :self.n_predictors * 5]
        target_classes = y_true[:, :, :, self.n_predictors * 5:]

        # calculate the IoU between the predicted and target boxes
        iou = compute_iou(pred_boxes[..., :4], target_boxes[..., :4])

        # create masks
        obj_mask = target_boxes[..., 4] > 0
        noobj_mask = target_boxes[..., 4] == 0

        # calculate the coordinate loss
        coord_loss = self.lambda_coord * torch.sum(obj_mask * (torch.sum((pred_boxes - target_boxes)**2, dim=-1)))

        # calculate the confidence loss
        pred_confidence = pred_boxes[..., 4]
        target_confidence = target_boxes[..., 4]
        confidence_loss_obj = torch.sum(obj_mask * (pred_confidence - iou)**2)
        confidence_loss_noobj = torch.sum(noobj_mask * (pred_confidence - target_confidence)**2)
        confidence_loss = confidence_loss_obj + self.lambda_noobj * confidence_loss_noobj

        # calculate the class loss
        class_loss = torch.sum(obj_mask * (pred_classes - target_classes)**2)

        # calculate the total loss
        loss = coord_loss + confidence_loss + class_loss

        return loss
