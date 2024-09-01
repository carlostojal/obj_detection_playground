import torch
from torch.utils.data import Dataset
from PIL import Image
import fiftyone as fo
from torchvision.transforms import transforms
from typing import Tuple

classes_dict = {
    "yellow_cone": 0,
    "blue_cone": 1,
    "orange_cone": 2,
    "large_orange_cone": 3,
    "unknown_cone": 4
}

class FSOCO_FiftyOne(Dataset):
    """
    FSOCO dataset loaded from FiftyOne.
    """

    def __init__(self, split: str, fiftyone_dataset: fo.Dataset,
                 img_width: int = 640, img_height: int = 480,
                 max_boxes: int = 30) -> None:
        """
        Initialize an instance of the FSOCO dataset.

        Args:
            split (str): Split of the dataset to load (train, val, test).
            fiftyone_name (str): Name of the FiftyOne dataset. Default is "fsoco".

        Returns:
            None
        """

        self.split = split
        self.dataset = fiftyone_dataset
        self.img_width = img_width
        self.img_height = img_height
        self.max_boxes = max_boxes

        # get a view of the samples of the split
        self.samples = self.dataset.match_tags(self.split)

        # get the sample IDs
        self.sample_ids = self.samples.values("id")

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """

        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Get the sample at the index.

        Args:
            idx (int): Index of the sample.

        Returns:
            (torch.Tensor): Image tensor shaped (num_channels, height, width).
            (torch.Tensor): Bounding boxes tensor shaped (max_boxes, 5) where the last dimension is [x, y, h, w, class].
        """

        # get the sample ID
        id = self.sample_ids[idx]
        sample = self.samples[id]

        # get the image
        img = Image.open(sample.filepath)
        img = img.convert("RGB")
        width = img.width
        height = img.height
        # resize the image to fit in the target size
        aspect_ratio = width / height
        if aspect_ratio > 1:
            width_ratio = width / self.img_width
            new_width = self.img_width
            new_height = int(height / width_ratio)
        else:
            height_ratio = height / self.img_height
            new_height = self.img_height
            new_width = int(width / height_ratio)
        img = img.resize((new_width, new_height))

        # pad the image
        pad_width = self.img_width - new_width
        pad_height = self.img_height - new_height
        img = transforms.Pad((int(pad_width/2), int(pad_height/2), int(pad_width/2), int(pad_height/2)))(img)
        
        # convert the image to tensor
        transform = transforms.ToTensor()
        img = transform(img)

        # create a tensor of zeros for the bounding boxes
        bboxes = torch.zeros((self.max_boxes, 5))

        # iterate the ground truth objects
        cur_box: int = 0
        for detection in sample['ground_truth']['detections']:
            bboxes[cur_box, :4] = torch.tensor(detection.bounding_box)
            bboxes[cur_box, 4] = classes_dict[detection.label]
        
        # convert the bounding boxes coordinates and dimensions to the new ratio
        bboxes[:, 0] += pad_width / 2
        bboxes[:, 1] += pad_height / 2
        bboxes[:, 2] *= new_width / width
        bboxes[:, 3] *= new_height / height

        return id, img, bboxes
