import torch
from typing import Tuple

def pad_image(img: torch.Tensor, bboxes: torch.Tensor, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad the image and bounding boxes to the target size.

    Args:
        img (torch.Tensor): Image tensor shaped (num_channels, height, width).
        bboxes (torch.Tensor): Bounding boxes tensor shaped (max_boxes, 5) where the last dimension is [x, y, h, w, class] normalized to image size.
        target_size (Tuple[int, int]): Target size of the image (height, width).

    Returns:
        (torch.Tensor) Padded image tensor shaped (num_channels, target_height, target_width).
        (torch.Tensor) Padded bounding boxes tensor shaped (max_boxes, 5) where the last dimension is [x, y, h, w, class] normalized to image size.
    """
    
    # resize the image
    width = img.size(2)
    height = img.size(1)
    aspect_ratio = width / height
    if aspect_ratio > 1:
        width_ratio = width / target_size[1]
        new_width = target_size[1]
        new_height = int(height / width_ratio)
    else:
        height_ratio = height / target_size[0]
        new_height = target_size[0]
        new_width = int(width / height_ratio)
    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)

    # pad the image
    pad_width = target_size[1] - new_width
    pad_height = target_size[0] - new_height
    img = torch.nn.functional.pad(img.unsqueeze(0), (int(pad_width/2), int(pad_height/2), int(pad_width/2), int(pad_height/2)), mode='constant', value=0).squeeze(0)

    # update the bounding boxes
    bboxes[:, 0] += (pad_width / 2) / new_width
    bboxes[:, 1] += (pad_height / 2) / new_height
    bboxes[:, 2] *= new_width / width
    bboxes[:, 3] *= new_height / height

    return img, bboxes

def unpad_bboxes(bboxes: torch.Tensor, padded_img_dims: Tuple[int, int], padding: Tuple[int, int]) -> torch.Tensor:
    """
    Unpad the bounding boxes to the original image size.

    Args:
        bboxes (torch.Tensor): Bounding boxes tensor shaped (max_boxes, 5) where the last dimension is [x, y, h, w, class] normalized to padded image size.
        padded_img_dims (Tuple[int, int]): Padded image dimensions (height, width).
        padding (Tuple[int, int]): Padding applied to the image (height, width) (total in both sides in pixels).

    Returns:
        (torch.Tensor) Unpadded bounding boxes tensor shaped (max_boxes, 5) where the last dimension is [x, y, h, w, class].
    """
    
    # get the image dimensions
    img_height, img_width = padded_img_dims

    # convert back to pixel values in the padded image
    bboxes[:, 0] *= img_width
    bboxes[:, 1] *= img_height
    bboxes[:, 2] *= img_height
    bboxes[:, 3] *= img_width

    # subtract the padding
    bboxes[:, 0] -= padding[1] / 2
    bboxes[:, 1] -= padding[0] / 2
    bboxes[:, 2] *= (img_height - padding[0]) / img_height
    bboxes[:, 3] *= (img_width - padding[1]) / img_width

    # scale the bounding boxes to the original image size
    bboxes[:, 0] /= img_width - padding[1]
    bboxes[:, 1] /= img_height - padding[0]
    bboxes[:, 2] /= img_height - padding[0]
    bboxes[:, 3] /= img_width - padding[1]

    return bboxes
