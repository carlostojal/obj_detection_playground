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

    def __init__(self, split: str = "train", fiftyone_name: str = "fsoco",
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
        self.fiftyone_name = fiftyone_name
        self.img_width = img_width
        self.img_height = img_height
        self.max_boxes = max_boxes

        # load the FiftyOne dataset
        self.dataset = fo.load_dataset(self.fiftyone_name)

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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the sample at the index.

        Args:
            idx (int): Index of the sample.

        Returns:
            (torch.Tensor): Image tensor shaped (num_channels, height, width).
            (torch.Tensor): Bounding boxes tensor shaped (max_boxes, 5) where the last dimension is [x, y, h, w, class].
        """

        # get the sample ID
        sample = self.samples[self.sample_ids[idx]]

        # get the image
        img = Image.open(sample.filepath)
        img = img.convert("RGB")
        img = img.resize((self.img_width, self.img_height))
        
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

        return img, bboxes
