from __future__ import absolute_import

import cv2
import numpy as np

from .colors import label_color


def draw_caption(image, point, caption):
    """ Draws a caption above the point in an image.
    # Arguments
        image   : The image to draw on.
        box     : A list of 2 elements in (x, y) format.
        caption : String containing the text to draw.
    """
    p = np.array(point[:2]).astype(int)
    cv2.putText(image, caption, (p[0], p[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (p[0], p[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_box(image, box, color=(255, 0, 0), thickness=2):
    """ Draws a box on an image with a given color.
    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_boxes(image, boxes, color=(255, 0, 0), thickness=2):
    """ Draws boxes on an image with a given color.
    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_point(image, point, color=(0, 255, 0)):
    """ Draws a point on an image
    # Arguments
        image : The image to draw on
        point : The (x, y) coordinate of the point
        color : The color of the points. Default is green
    """
    p = np.array(point[:2]).astype(int)
    cv2.circle(image, (p[0], p[1]), 2, color, -1)


def draw_points(image, points, color=(0, 255, 0)):
    """ Draws points on an image
    # Arguments
        image  : The image to draw on
        points : A (num_points, 2) shaped array containing the (x, y) coordinates of the points
        color  : The color of the points. Default is green
    """
    for point in points:
        draw_point(image, point, color=color)


def draw_mask(image, mask, color=(0, 0, 255), alpha=0.3):
    """ Draws a mask in an image
    # Arguments
        image : The image to draw on
        mask  : The mask which should be the same size as image but only single channel
        color : The color of the mask
        alpha : degree of visibility of mask
    """
    mask = np.array(mask, dtype='float32')

    if len(mask.shape) == 3:
        assert mask.shape[-1] == 1
        mask = mask[:, :, 0]
    assert len(mask.shape) == 2

    mask_pos = mask > 0
    mask = [mask * c * alpha for c in color]
    mask = np.stack(mask, axis=2).astype('uint8')
    image[mask_pos] = ((image[mask_pos]).astype('float32') * (1 - alpha)).astype('uint8')
    image[mask_pos] += mask[mask_pos]


def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.
    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_pipeline.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = '{} : {:.2f}'.format(
            label_to_name(labels[i]) if label_to_name else labels[i],
            scores[i]
        )
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=None, label_to_name=None):
    """ Draws annotations in an image.
    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label).
        color         : The color of the boxes. By default the color from keras_pipeline.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    for a in annotations:
        label   = a[4]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, a, caption)

        draw_box(image, a, color=c)
