"""
Script containing functions to transform data generated by image datasets in
kindler.data.datasets
"""
from __future__ import absolute_import
from __future__ import division

import random
import logging

import cv2
import numpy as np

from torch._six import string_classes, container_abcs

logger = logging.getLogger(__name__)


def check_image_is_numpy(image):
    assert isinstance(image, np.ndarray)
    assert len(image.shape) == 3


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.RandomHorizontalFlip(),
        >>>     transforms.ImageNormalization(),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ImageResize(object):
    """ Add Class helper """
    def __init__(
        self,
        height=None,
        width=None,
        min_size=None,
        max_size=None
    ):
        """ Add args helper """
        if height is not None:
            assert width is not None, \
                'If height is provided, then width must be provided as well'
            if min_size is not None or max_size is not None:
                logger.warn('height and width are already provided, min_size and max_size will be ignored')

            self.height = height
            self.width = width
            self.mode = 'fixed'

        else:
            assert min_size is not None, \
                'Please provide either height and width or min_size'

            self.min_size = min_size
            self.max_size = max_size
            self.mode = 'flex'

    def _determine_size(self, image_shape):
        orig_height, orig_width = image_shape[:2]

        if self.mode == 'fixed':
            # Return static height width if fixed mode
            height_scale = self.height / orig_height
            width_scale = self.width / orig_width
            return self.height, self.width, height_scale, width_scale

        # Assume flex mode
        min_orig_size = min(orig_height, orig_width)
        max_orig_size = max(orig_height, orig_width)

        # Attemp scaling at min_size and verify if it satisfies max_size
        # If max_size is not satisfied, use scaling at max_size
        scale = self.min_size / min_orig_size
        if self.max_size is not None:
            if self.max_size > max_orig_size * scale:
                scale = self.max_size / max_orig_size

        new_height = round(orig_height * scale)
        new_width = round(orig_width * scale)

        return new_height, new_width, scale, scale

    def __call__(self, item):
        check_image_is_numpy(item['image'])
        height, width, height_scale, width_scale = self._determine_size(item['image'].shape)
        item['image'] = cv2.resize(item['image'], (width, height))

        if 'annotations' in item:
            item['annotations'][:, 0::2] *= width_scale
            item['annotations'][:, 1::2] *= height_scale

        if 'masks' in item:
            masks = []
            for mask in item['masks']:
                mask = cv2.resize(mask, (width, height))
                masks.append(mask)
            item['masks'] = np.array(masks)

        return item


class RandomHorizontalFlip(object):
    """ Add Class helper """
    def __init__(self, prob=0.5):
        """ Add args helper """
        self.prob = prob

    def __call__(self, item):
        check_image_is_numpy(item['image'])
        if random.random() > self.prob:
            return item

        item['image'] = item['image'][:, ::-1]
        image_width = item['image'].shape[1]

        if 'annotations' in item:
            temp = item['annotations'][:, 0].copy()
            item['annotations'][:, 0] = image_width - item['annotations'][:, 2]
            item['annotations'][:, 2] = image_width - temp

        if 'masks' in item:
            item['masks'] = item['masks'][:, :, ::-1]

        return item


class RandomVerticalFlip(object):
    """ Add Class helper """
    def __init__(self, prob=0.5):
        """ Add args helper """
        self.prob = prob

    def __call__(self, item):
        check_image_is_numpy(item['image'])
        if random.random() > self.prob:
            return item

        item['image'] = item['image'][::-1, :]
        image_height = item['image'].shape[0]

        if 'annotations' in item:
            temp = item['annotations'][:, 1].copy()
            item['annotations'][:, 1] = image_height - item['annotations'][:, 3]
            item['annotations'][:, 3] = image_height - temp

        if 'masks' in item:
            item['masks'] = item['masks'][:, ::-1, :]

        return item


class ImageNormalization(object):
    """ Add Class helper """
    def __init__(self):
        """
        There is no mean or std settings available
        All torchvision models are trained using VGG normalization
        There aren't any signs that things will change from this.
        """
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]

    def __call__(self, item):
        check_image_is_numpy(item['image'])
        item['image'][..., 0] = (item['image'][..., 0] - self.mean[0]) / self.std[0]
        item['image'][..., 1] = (item['image'][..., 1] - self.mean[1]) / self.std[1]
        item['image'][..., 2] = (item['image'][..., 2] - self.mean[2]) / self.std[2]
        return item


class ToTensor(object):
    """ Add Class helper """
    def __init__(self):
        pass

    @classmethod
    def _to_tensor(cls, obj):
        pass
        if isinstance(obj, np.ndarray):
            return np.FloatTensor(obj)
        elif isinstance(obj, string_classes):
            return obj
        elif isinstance(obj, container_abcs.Mapping):
            return {k: cls.to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, container_abcs.Sequence):
            return [cls.to_tensor(e) for e in obj]
        else
            return obj

    def __call__(self, item):
        item = self._to_tensor(item)
        item['image'] = item['image'].transpose(2, 0, 1) / 255
        return item
