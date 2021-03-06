"""
Script to store anchor related functions
that uses torch.Tensor
"""
from __future__ import division

import torch


def meshgrid2d(x, y):
    xx = x.repeat(len(y), 1)
    yy = y.repeat(len(x), 1).permute(1, 0)
    return xx, yy


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) Tensor
    b: (K, 4) Tensor
    Returns
    -------
    overlaps: (N, K) Tensor of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(a[:, 2:3], b[:, 2]) - torch.max(a[:, 0:1], b[:, 0])
    ih = torch.min(a[:, 3:4], b[:, 3]) - torch.max(a[:, 1:2], b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = (a[:, 2:3] - a[:, 0:1]) * (a[:, 3:4] - a[:, 1:2]) + area - iw * ih
    ua = torch.clamp(ua, min=1e-15)

    intersection = iw * ih

    return intersection / ua


def generate_anchors_at_window(
    base_size=32,
    ratios=[0.5, 1., 2.],
    scales=[2. ** 0., 2. ** (1./3.), 2. ** (2./3.)]
):
    """ Generate anchors based on a size a set of ratios and scales
    w.r.t a reference window
    """
    # if not isinstance(base_size, torch.Tensor):
    #     base_size = torch.Tensor([base_size]).reshape(1)
    if not isinstance(ratios, torch.Tensor):
        ratios = torch.Tensor(ratios)
    if not isinstance(scales, torch.Tensor):
        scales = torch.Tensor(scales)

    num_ratios = len(ratios)
    num_scales = len(scales)
    num_anchors = num_ratios * num_scales
    tiled_scales = scales.repeat(num_ratios)
    repeated_ratios = torch.stack([ratios] * num_scales).transpose(0, 1).reshape(-1)

    # initialize output anchors
    anchors = torch.zeros(num_anchors, 4)
    anchors[:, 2] = base_size * tiled_scales
    anchors[:, 3] = base_size * tiled_scales

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = torch.sqrt(areas / repeated_ratios)
    anchors[:, 3] = anchors[:, 2].clone() * repeated_ratios

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] = anchors[:, 0::2].clone() - anchors[:, 2:3].clone() / 2
    anchors[:, 1::2] = anchors[:, 1::2].clone() - anchors[:, 3:4].clone() / 2

    return anchors


def shift_anchors(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size """
    shift_x = torch.arange(0 + 0.5, shape[1] + 0.5, step=1) * stride
    shift_y = torch.arange(0 + 0.5, shape[0] + 0.5, step=1) * stride
    if anchors.is_cuda:
        device_idx = torch.cuda.device_of(anchors).idx
        shift_x = shift_x.cuda(device_idx)
        shift_y = shift_y.cuda(device_idx)

    shift_x, shift_y = meshgrid2d(shift_x, shift_y)

    shifts = torch.stack([
        shift_x.reshape(-1), shift_y.reshape(-1),
        shift_x.reshape(-1), shift_y.reshape(-1)
    ], dim=1)

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape(1, A, 4) + shifts.reshape(1, K, 4).permute(1, 0, 2)
    all_anchors = all_anchors.reshape(K * A, 4)

    return all_anchors


def bbox_transform(anchors, gt_boxes, mean=0.0, std=0.2):
    """ Compute bounding-box regression targets for an image """
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    targets = torch.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2), dim=1)
    targets = (targets - mean) / std

    return targets


def bbox_transform_inv(boxes, deltas, mean=0.0, std=0.2):
    """ Applies deltas (usually regression results) to boxes (usually anchors).
    Before applying the deltas to the boxes, the normalization that was
    previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are
    unnormalized in this function and then applied to the boxes.
    Args
        boxes : torch.Tensor of shape (N, 4), where N the number of boxes and
            4 values for (x1, y1, x2, y2).
        deltas: torch.Tensor of same shape as boxes. These deltas
            (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean  : The mean value used when computing deltas
            (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas
            (defaults to [0.2, 0.2, 0.2, 0.2]).
    Returns
        A torch.Tensor of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    width = boxes[..., 2] - boxes[..., 0]
    height = boxes[:, :, 3] - boxes[..., 1]

    x1 = boxes[..., 0] + (deltas[..., 0] * std + mean) * width
    y1 = boxes[..., 1] + (deltas[..., 1] * std + mean) * height
    x2 = boxes[..., 2] + (deltas[..., 2] * std + mean) * width
    y2 = boxes[..., 3] + (deltas[..., 3] * std + mean) * height

    pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    return pred_boxes


def anchor_targets_bbox(
    anchors,
    annotations,
    num_classes,
    mask_shape=None,
    use_class_specific_bbox=False,
    positive_overlap=0.5,
    negative_overlap=0.4
):
    """ Generate anchor targets for bbox detection.
    Args
        anchors: torch.Tensor of shape (A, 4) in the (x1, y1, x2, y2) format.
        annotations: torch.Tensor of shape (N, 5) in the
            (x1, y1, x2, y2, label) format.
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used
            to mark the relevant part of the image.
        use_class_specific_bbox: Should each class have it's own bbox?
        negative_overlap: IoU overlap for negative anchors
            (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors
            (all anchors with overlap > positive_overlap are positive).
    Returns
        cls_target: torch.Tensor containing the classification target at
            each anchor position shape will be (A, num_classes)
        bbox_target: torch.Tensor containing the detection bbox at each
            anchor position shape will be (A, 4) if not using class specific
            bbox or (A, 4 * num_classes) if using class specific bbox
        anchor_states: anchor_states: torch.Tensor of shape (N,) containing
            the state of each anchor (-1 for ignore, 0 for bg, 1 for fg).
    """
    # Create blobs that will hold results
    # anchor states: 1 is positive, 0 is negative, -1 is dont care
    anchor_states = torch.zeros_like(anchors[:, 0])
    cls_target = torch.stack([anchor_states] * num_classes, dim=1)
    if use_class_specific_bbox:
        bbox_target = torch.stack([anchor_states] * 4 * num_classes, dim=1)
    else:
        bbox_target = torch.stack([anchor_states] * 4, dim=1)

    if annotations.shape[0] > 0:
        # obtain indices of gt annotations with the greatest overlap
        overlaps = compute_overlap(anchors, annotations)
        argmax_overlaps_inds = torch.argmax(overlaps, dim=1)
        max_overlaps = overlaps[range(overlaps.shape[0]), argmax_overlaps_inds]

        # assign "dont care" labels
        ignore_inds = max_overlaps > negative_overlap
        anchor_states[ignore_inds] = -1
        positive_inds = max_overlaps >= positive_overlap
        anchor_states[positive_inds] = 1

        # compute classification and regression targets
        total_positive_inds = torch.sum(positive_inds)
        if total_positive_inds > 0:
            annotations = annotations[argmax_overlaps_inds[positive_inds]]

            # Scatter classification and regression targets to positive indices
            # The current implementation is very naive and inefficient
            # though the overall cost in the scope of training is rather
            # insignificant
            # Possible improvements with scatter functions
            positive_cls = annotations[:, 4].long()
            positive_bbox = annotations[:, :4]
            positive_cls_target = cls_target[positive_inds]
            positive_bbox_target = bbox_target[positive_inds]

            for i in range(total_positive_inds):
                positive_cls_target[i, positive_cls[i]] = 1
                if use_class_specific_bbox:
                    bbox_start_pos = positive_cls[i] * 4
                    positive_bbox_target[i, bbox_start_pos:bbox_start_pos+4] = positive_bbox[i]
                else:
                    positive_bbox_target[i] = positive_bbox[i]

            cls_target[positive_inds] = positive_cls_target
            bbox_target[positive_inds] = positive_bbox_target

    # ignore annotations outside of image
    if mask_shape is not None:
        anchors_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        inds = (anchors_centers[:, 0] >= mask_shape[-1]) | (anchors_centers[:, 1] >= mask_shape[-2])
        anchor_states[inds] = -1

    return cls_target, bbox_target, anchor_states
