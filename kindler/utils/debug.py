"""
Script containing some functions for debugging purposes
"""
from __future__ import absolute_import
from __future__ import division

from PIL import Image
import numpy as np
import torch

from . import vis, colors

VGG_MEAN = [123.675, 116.28, 103.53]
VGG_STD = [58.395, 57.12, 57.375]


def tensor_to_np(tensor):
    return tensor.detach().cpu().data.numpy()


def tensor_to_image(tensor_image):
    """ Converts a preprocessed image tensor back into a PIL Image """
    assert isinstance(tensor_image, torch.Tensor)
    assert len(tensor_image.shape) == 3
    assert tensor_image.shape[0] == 3

    np_image = tensor_to_np(tensor_image)
    np_image = np_image.transpose(1, 2, 0) * 255
    np_image[..., 0] = np_image[..., 0] * VGG_STD[0] + VGG_MEAN[0]
    np_image[..., 1] = np_image[..., 1] * VGG_STD[1] + VGG_MEAN[1]
    np_image[..., 2] = np_image[..., 2] * VGG_STD[2] + VGG_MEAN[2]

    np_image = np_image.round().clip(min=0, max=255).astype('uint8')
    return Image.fromarray(np_image)

def visualize_batch(batch):
    """ Visualize a collated image batch """
    batch_size = len(batch['image'])
    for i in range(batch_size):
        image = tensor_to_image(batch['image'][i])
        image = np.array(image).astype('float32').copy()

        if 'annotations' in batch:
            anns = tensor_to_np(batch['annotations'][i])
            vis.draw_annotations(image, anns)

        if 'masks' in batch:
            anns = tensor_to_np(batch['annotations'][i])
            print(anns)
            ann_cls = anns[:, 4].astype(int)
            masks = tensor_to_np(batch['masks'][i])
            for cls, mask in zip(ann_cls, masks):
                vis.draw_mask(image, mask, color=colors.label_color(cls))

        image = Image.fromarray(image.astype('uint8'))
        image.save('debug_image_batch_{}.jpg'.format(i))
