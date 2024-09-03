import torch
from typing import Tuple
from math import floor

def pad_image(img: torch.Tensor, bboxes: torch.Tensor, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Pad the image and bounding boxes to the target size.

    Args:
        img (torch.Tensor): Image tensor shaped (num_channels, height, width).
        bboxes (torch.Tensor): Bounding boxes tensor shaped (max_boxes, 5) where the last dimension is [x, y, h, w, class] normalized to image size.
        target_size (Tuple[int, int]): Target size of the image (height, width).

    Returns:
        (torch.Tensor) Padded image tensor shaped (num_channels, target_height, target_width).
        (torch.Tensor) Padded bounding boxes tensor shaped (max_boxes, 5) where the last dimension is [x, y, h, w, class] normalized to image size.
        (Tuple[int, int]) Padding applied to the image (height, width) (total in both sides in pixels).
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
    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

    # pad the image
    pad_width = target_size[1] - new_width
    pad_height = target_size[0] - new_height
    pad_width_sym = pad_width / 2
    pad_height_sym = pad_height / 2
    img = torch.nn.functional.pad(img, (int(pad_width_sym), int(pad_width_sym), int(pad_height_sym), int(pad_height_sym)), mode='constant', value=0)
    # add rows or columns if the padding is not even
    if pad_width % 2 != 0:
        img = torch.nn.functional.pad(img, (0, 1), mode='constant', value=0) # pad on right
    if pad_height % 2 != 0:
        img = torch.nn.functional.pad(img, (0, 0, 0, 1), mode='constant', value=0) # pad on bottom

    # the new_width, new_height par are the unpadded resized image dimensions
    # calculate the new padded dimensions
    new_width_padded = new_width + pad_width
    new_height_padded = new_height + pad_height

    # convert to pixel coordinates
    bboxes[:, 0] *= new_width
    bboxes[:, 1] *= new_height
    bboxes[:, 2] *= new_height 
    bboxes[:, 3] *= new_width

    # add the padding to the boundin coordinates
    bboxes[:, 0] += pad_width / 2
    bboxes[:, 1] += pad_height / 2

    # normalize
    bboxes[:, 0] /= new_width_padded
    bboxes[:, 1] /= new_height_padded
    bboxes[:, 2] /= new_height_padded
    bboxes[:, 3] /= new_width_padded

    return img, bboxes, (pad_height, pad_width)

def unpad_bboxes(bboxes: torch.Tensor, padded_img_dims: Tuple[int, int], padding: Tuple[int, int]) -> torch.Tensor:
    """
    Unpad the bounding boxes to the original image size.

    Args:
        bboxes (torch.Tensor): Bounding boxes tensor shaped (max_boxes, 5) where the last dimension is [x, y, h, w, class] normalized to padded image size.
        padded_img_dims (Tuple[int, int]): Padded image dimensions (height, width).
        padding (Tuple[int, int]): Padding applied to the image (height, width) (total in both sides in pixels, in the resized image).

    Returns:
        (torch.Tensor) Unpadded bounding boxes tensor shaped (max_boxes, 5) where the last dimension is [x, y, h, w, class].
    """
    
    # get the image dimensions
    height, width = padded_img_dims

    # get the image dimensions without padding
    new_width = width - padding[1]
    new_height = height - padding[0]

    # convert to pixel coordinates
    bboxes[:, 0] *= width
    bboxes[:, 1] *= height
    bboxes[:, 2] *= height
    bboxes[:, 3] *= width

    # remove the padding from the bounding box coordinates
    bboxes[:, 0] -= padding[1] / 2
    bboxes[:, 1] -= padding[0] / 2

    # convert to normalized values in the unpadded image
    bboxes[:, 0] /= new_width
    bboxes[:, 1] /= new_height
    bboxes[:, 2] /= new_height
    bboxes[:, 3] /= new_width

    return bboxes
