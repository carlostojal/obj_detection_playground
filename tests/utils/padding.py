import unittest
import torch
import sys
sys.path.append('.')
from datasets.utils import pad_image, unpad_bboxes

class TestPadding(unittest.TestCase):

    def test_pad_image(self):
        img = torch.rand(3, 1000, 2000)
        bboxes = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0], [0.2, 0.2, 0.2, 0.2, 1]])
        target_size = (480, 640)
        padded_img, padded_bboxes, padding = pad_image(img, bboxes, target_size)
        self.assertEqual(padded_img.size(), (3, 480, 640))
        self.assertEqual(padding, (160, 0))

    def test_pad_bboxes(self):
        img = torch.rand(3, 1000, 2000)
        bboxes = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0], [0.2, 0.2, 0.2, 0.2, 1]])
        target_size = (480, 640)
        padded_img, padded_bboxes, padding = pad_image(img, bboxes, target_size)
        self.assertEqual(padded_bboxes.size(), (2, 5))
        self.assertEqual(padded_bboxes[0, 0], 0.1)
        self.assertAlmostEqual(padded_bboxes[0, 1], 7/30)
        self.assertAlmostEqual(padded_bboxes[0, 2], 2/30)
        self.assertAlmostEqual(padded_bboxes[0, 3], 0.1)
