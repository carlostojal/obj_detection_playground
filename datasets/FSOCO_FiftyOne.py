import torch
from torch.utils.data import Dataset
from PIL import Image
import fiftyone as fo
from typing import Tuple

class FSOCO_FiftyOne(Dataset):
    """
    FSOCO dataset loaded from FiftyOne.
    """

    def __init__(self, split: str = "train", fiftyone_name: str = "fsoco") -> None:
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

        # load the FiftyOne dataset
        self.dataset = fo.load_dataset(self.fiftyone_name)

        # get a view of the samples of the split
        self.samples = self.dataset.match_tags(self.split)

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
            (torch.Tensor): Image tensor shaped (batch_size, num_channels, height, width).
            (torch.Tensor): Bounding boxes tensor shaped (batch_size, n_boxes, 5) where the last dimension is [x, y, h, w, class].
        """

        # TODO
