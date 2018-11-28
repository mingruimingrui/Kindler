"""
Helper functions to work with masks
"""
import cv2
import numpy as np
from pycocotools import mask as maskUtils


def segm_to_mask(segm, h=None, w=None):
    """
    Converts a segmentation which can be polygon, uncompressed RLE to RLE
    """
    rle = segm_to_rle(segm, h, w)
    return maskUtils.decode(rle)


def segm_to_rle(segm, h=None, w=None):
    """
    Converts a segmentation into RLE
    Segmentation can be polygons, uncompressed RLE or RLE
    """
    if isinstance(segm, list):
        # polygons
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        rle = segm

    return rle


def mask_to_polygon(mask):
    """
    Converts a mask into a polygon segmentation
    """
    # Make a copy of mask of type uint8
    mask = np.array(mask, dtype='uint8').copy()

    # Compute contours
    _, contours, _ = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Retrieve segmentations
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        segmentation.append(contour)

    return segmentation
