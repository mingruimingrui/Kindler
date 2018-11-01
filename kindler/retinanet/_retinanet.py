"""
Script used to store RetinaNet misc
"""
from __future__ import division

import math
import torch
from ..utils import anchors as utils_anchors

conv_3x3_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
conv_1x1_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0}


class ClassificationHead(torch.nn.Module):
    """
    Simple fully convolutional classifier
    """
    def __init__(
        self,
        input_feature_size,
        feature_size,
        num_layers,
        num_anchors,
        num_classes,
        use_background_predictor=False,
        prior_prob=0.01
    ):
        """
        Args:
            input_feature_size: The network input feature size
            feature_size: The internal network feature size
            num_layers: The number of convolutional layers before classifier
            num_anchors: The number of anchors per moving window
            num_classes: The number of classes to predict
            use_background_predictor: Should there be a classifier for background?
            prior_prob: The initial classifier output value
        """
        super(ClassificationHead, self).__init__()

        self.total_num_classes = num_classes
        if use_background_predictor:
            self.total_num_classes += 1

        # Add conv head
        head = []
        for i in range(num_layers):
            head.append(torch.nn.Conv2d(
                input_feature_size if i == 0 else feature_size,
                feature_size,
                **conv_3x3_kwargs
            ))
            head.append(torch.nn.ReLU(inplace=True))

        # Add classifier layer
        head.append(torch.nn.Conv2d(
            input_feature_size if num_layers == 0 else feature_size,
            self.total_num_classes * num_anchors,
            **conv_1x1_kwargs
        ))
        head.append(torch.nn.Sigmoid())

        # Initialize classification output to prior_prob
        # kernel ~ 0.0
        # bias   ~ -log((1 - prior_prob) / prior_prob)  So that output is prior_prob after sigmoid
        kernel = head[-2].weight
        bias = head[-2].bias
        kernel.data.fill_(0.0)
        bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))

        self.head = torch.nn.Sequential(*head)

    def forward(self, x):
        x = self.head(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.total_num_classes)


class RegressionHead(torch.nn.Module):
    """
    Simple fully convolutional regressor
    """
    def __init__(
        self,
        input_feature_size,
        feature_size,
        num_layers,
        num_anchors,
        num_classes=None,
        use_class_specific_bbox=False
    ):
        """
        num_classes is only needed if use_class_specific_bbox is True

        Args:
            input_feature_size: The network input feature size
            feature_size: The internal network feature size
            num_layers: The number of convolutional layers before classifier
            num_anchors: The number of anchors per moving window
            num_classes: The number of classes to predict
            use_class_specific_bbox: Should every class have it's own bbox regressor?
        """
        super(RegressionHead, self).__init__()

        self.total_num_bbox = 4
        if use_class_specific_bbox:
            assert num_classes is not None
            self.total_num_bbox *= num_classes

        # Add conv head
        head = []
        for i in range(num_layers):
            head.append(torch.nn.Conv2d(
                input_feature_size if i == 0 else feature_size,
                feature_size,
                **conv_3x3_kwargs
            ))
            head.append(torch.nn.ReLU(inplace=True))

        # Add regression layer
        head.append(torch.nn.Conv2d(
            input_feature_size if num_layers == 0 else feature_size,
            num_anchors * self.total_num_bbox,
            **conv_1x1_kwargs
        ))

        self.head = torch.nn.Sequential(*head)

    def forward(self, x):
        x = self.head(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.total_num_bbox)


class CombinedHead(torch.nn.Module):
    """
    Simple fully convolutional model that outputs both classification and
    regression results
    """
    def __init__(
        self,
        input_feature_size,
        feature_size,
        num_layers,
        num_anchors,
        num_classes,
        use_class_specific_bbox=False,
        use_background_predictor=False,
        prior_prob=0.01
    ):
        """
        Args:
            input_feature_size: The network input feature size
            feature_size: The internal network feature size
            num_layers: The number of convolutional layers before classifier
            num_anchors: The number of anchors per moving window
            num_classes: The number of classes to predict
            use_class_specific_bbox: Should every class have it's own bbox regressor?
            use_background_predictor: Should there be a classifier for background?
            prior_prob: The initial classifier output value
        """
        super(CombinedHead, self).__init__()

        self.total_num_classes = num_classes
        if use_background_predictor:
            self.total_num_classes += 1

        self.total_num_bbox = 4
        if use_class_specific_bbox:
            assert num_classes is not None
            self.total_num_bbox *= num_classes

        self.split_point = num_anchors * self.total_num_classes

        # Add conv head
        head = []
        for i in range(num_layers):
            head.append(torch.nn.Conv2d(
                input_feature_size if i == 0 else feature_size,
                feature_size,
                **conv_3x3_kwargs
            ))
            head.append(torch.nn.ReLU(inplace=True))

        # Add combined layer
        head.append(torch.nn.Conv2d(
            input_feature_size if num_layers == 0 else feature_size,
            num_anchors * (self.total_num_classes + self.total_num_bbox),
            **conv_1x1_kwargs
        ))

        # Initialize classification output to prior_prob
        # kernel ~ 0.0
        # bias   ~ -log((1 - prior_prob) / prior_prob)  So that output is prior_prob after sigmoid
        kernel = head[-1].weight[:self.split_point]
        bias = head[-1].bias[:self.split_point]
        # kernel = head[-1].weight
        # bias = head[-1].bias
        kernel.data.fill_(0.0)
        bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))

        self.head = torch.nn.Sequential(*head)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.head(x)

        classification = x[:, :self.split_point].permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.total_num_classes)
        regression = x[:, self.split_point:].permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.total_num_bbox)

        return self.sigmoid(classification), regression


class Anchor(torch.nn.Module):
    def __init__(
        self,
        ratios=[0.5, 1., 2.],
        scales=[2. ** 0., 2. ** (1./3.), 2. ** (2./3.)],
        size=32,
        stride=8,
    ):
        super(Anchor, self).__init__()
        self.stride = stride
        self.anchors = utils_anchors.generate_anchors_at_window(
            base_size=size,
            ratios=ratios,
            scales=scales,
        )
        self.anchors = torch.nn.Parameter(self.anchors)
        self.anchors.requires_grad = False

    def forward(self, feature_shape):
        return utils_anchors.shift_anchors(feature_shape, self.stride, self.anchors)


class ComputeAnchors(torch.nn.Module):
    def __init__(
        self,
        feature_levels,
        ratios=[0.5, 1., 2.],
        scales_per_octave=3,
        size_mult=4.0,
        stride_mult=1.0,
    ):
        super(ComputeAnchors, self).__init__()

        scales = [2 ** (i / scales_per_octave) for i in range(scales_per_octave)]

        # Generate the Anchor layers which would compute the anchors at each
        # feature level
        anchor = dict()
        for level in feature_levels:
            assert isinstance(level, int)

            size = size_mult * (2 ** level)
            stride = stride_mult * (2 ** level)

            anchor[str(level)] = Anchor(
                ratios=ratios,
                scales=scales,
                size=size,
                stride=stride
            )

        self.anchor = torch.nn.ModuleDict(anchor)

    def forward(self, features):
        anchors = {}

        for level, feature in features.items():
            anchors[level] = self.anchor[str(level)](feature.shape[-2:])

        return anchors


class ComputeTargets(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        use_class_specific_bbox=False,
        use_background_predictor=False,
        regression_mean=0.0,
        regression_std=0.2,
        positive_overlap=0.5,
        negative_overlap=0.4
    ):
        super(ComputeTargets, self).__init__()
        self.num_classes = num_classes
        self.use_class_specific_bbox = use_class_specific_bbox
        self.use_background_predictor = use_background_predictor
        self.regression_mean = regression_mean
        self.regression_std = regression_std
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap

    def forward(self, annotations_batch, anchors):
        # Create blobs to store anchor informations
        cls_batch = []
        reg_batch = []
        states_batch = []

        for annotations in annotations_batch:
            cls_targets, bbox_targets, anchor_states = utils_anchors.anchor_targets_bbox(
                anchors=anchors,
                annotations=annotations,
                num_classes=self.num_classes,
                use_class_specific_bbox=self.use_class_specific_bbox,
                positive_overlap=self.positive_overlap,
                negative_overlap=self.negative_overlap
            )
            reg_targets = utils_anchors.bbox_transform(
                anchors=anchors,
                gt_boxes=bbox_targets,
                mean=self.regression_mean,
                std=self.regression_std
            )

            cls_batch.append(cls_targets)
            reg_batch.append(reg_targets)
            states_batch.append(anchor_states)

        cls_batch = torch.stack(cls_batch, dim=0)
        reg_batch = torch.stack(reg_batch, dim=0)
        states_batch = torch.stack(states_batch, dim=0)

        return cls_batch, reg_batch, states_batch
