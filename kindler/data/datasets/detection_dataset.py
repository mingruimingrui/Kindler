"""
Dataset for general detection based tasks
also covers segmentation and keypoint
"""

from __future__ import absolute_import
from __future__ import division

import os
import tqdm

import numpy as np

import torch
import torch.utils.data

from ...utils import mask as utils_mask
from ...utils.image_io import read_image, get_image_size


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_files,
        annotations,
        segms=None,
        keypoints=None,
        root_dir='/',
        transforms=None,
        image_heights=None,
        image_widths=None
    ):
        """
        Write documentations for this class
        """
        self.image_files = image_files
        self.annotations = annotations
        self.root_dir = root_dir

        if segms is not None:
            self.segms = segms

        if keypoints is not None:
            raise NotImplementedError('keypoints has not been implemented')
            self.keypoints = keypoints

        if transforms is not None:
            self.transforms = transforms

        if image_heights is not None and image_widths is not None:
            self.image_heights = image_heights
            self.image_widths = image_widths

        self._check_inputs()
        if segms is not None:
            self._convert_segms_to_rles()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        item = {
            'idx': idx,
            'image': np.asarray(read_image(image_path)),
            'annotations': np.array(self.annotations[idx])
        }
        if len(item['annotations']) == 0:
            item['annotations'] = np.zeros((0, 5), dtype='float32')

        if hasattr(self, 'segms'):
            w, h = item['image'].size
            segms = self.segms[idx]
            if len(segms) > 0:
                item['masks'] = [utils_mask.segm_to_mask(segm, h, w) for segm in segms]
            else:
                item['masks'] = np.zeros((0, h, w), dtype='float32')

        if hasattr(self, 'keypoints'):
            item['keypoints'] = self.keypoints[idx]

        # Assert all values float32
        # Essentially also makes a copy of each value so original does not get
        # editted
        for key, value in item.items():
            if key == 'idx':
                continue
            if key == 'annotations' and len(value) == 0:
                value = np.zeros((0, 5), dtype='float32')
            if key == 'segms':
                item[key] = np.array(value, dtype='float32')

        if hasattr(self, 'transforms'):
            item = self.transforms(item)

        return item

    def get_item_num_annotations(self, idx):
        return len(self.annotations[idx])

    def get_item_aspect_ratio(self, idx):
        self._auto_fill_image_heights_widths()
        return self.image_heights[idx] / self.image_widths[idx]

    def _check_inputs(self):
        """ Ensure that all inputs are valid """
        assert isinstance(self.image_files, list)
        assert isinstance(self.annotations, list)
        assert isinstance(self.root_dir, str)

        dataset_size = self.__len__()
        assert len(self.annotations) == dataset_size

        if hasattr(self, 'segms'):
            assert isinstance(self.segms, list)
            assert len(self.segms) == dataset_size

        if hasattr(self, 'keypoints'):
            assert isinstance(self.keypoints, list)
            assert len(self.keypoints) == dataset_size

        if hasattr(self, 'image_heights'):
            assert isinstance(self.image_heights, list)
            assert isinstance(self.image_widths, list)
            assert len(self.image_heights) == dataset_size
            assert len(self.image_widths) == dataset_size

        for f in self.image_files:
            assert os.path.isfile(os.path.join(self.root_dir, f))

    def _auto_fill_image_heights_widths(self):
        if hasattr(self, 'image_heights'):
            return

        desc = 'Profiling image sizes'
        image_sizes = []
        for f in tqdm.tqdm(self.image_files, desc=desc):
            image_sizes.append(get_image_size(f))
        self.image_heights, self.image_widths = zip(*image_sizes)
        self.image_heights = list(self.image_heights)
        self.image_widths = list(self.image_widths)

    def _convert_segms_to_rles(self):
        self._auto_fill_image_heights_widths()
        desc = 'Converting segms to rle'
        for i in tqdm.tqdm(range(self.__len__()), desc=desc):
            segms = []
            h = self.image_heights[i]
            w = self.image_widths[i]
            for segm in self.segms[i]:
                segms.append(utils_mask.segm_to_rle(segm, h, w))
            self.segms[i] = segms

    @staticmethod
    def mask_to_bbox(mask):
        """ Transforms a mask into bounding box in the xyxy format """
        img_h, img_w = mask.shape[:2]

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        x1 = int(max(cmin - 1, 0))
        y1 = int(max(rmin - 1, 0))
        x2 = int(min(cmax + 1, img_w))
        y2 = int(min(rmax + 1, img_h))

        return x1, y1, x2, y2

    @staticmethod
    def mask_to_segm(mask):
        """
        Converts a mask into a polygon segmentation
        Significantly this format can use used in the coco dataset
        Method exposed from kindler.utils.mask
        """
        return utils_mask.mask_to_polygon(mask)

    @staticmethod
    def get_image_size(image_file):
        """
        Get the image height and width from an image file without reading the
        entire file Method exposed from kindler.utils.image_io
        """
        return get_image_size(image_file)

    @staticmethod
    def xywh_to_xyxy(boxes, inplace=False):
        """ Converts boxes of final dimension 4 of type xywh to xyxy """
        if not inplace:
            boxes = boxes.copy()
        boxes[..., 2] += boxes[..., 0]
        boxes[..., 3] += boxes[..., 1]
        return boxes

    @staticmethod
    def xyxy_to_xywh(boxes, inplace=False):
        """ Converts boxes of final dimension 4 of type xyxy to xywh """
        if not inplace:
            boxes = boxes.copy()
        boxes[..., 2] -= boxes[..., 0]
        boxes[..., 3] -= boxes[..., 1]
        return boxes
