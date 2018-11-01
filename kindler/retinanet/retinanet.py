import torch

from .config import config, validate_config
from ..backbone import Backbone
from ..fpn import FPN
from ._retinanet import (
    ClassificationHead,
    RegressionHead,
    CombinedHead,
    ComputeAnchors,
    ComputeTargets
)


class RetinaNet(torch.nn.Module):
    def __init__(self, config_file=None, num_classes=None, **kwargs):
        super(RetinaNet, self).__init__()

        # Edit default configs if num_classes is provided
        config_ = config.clone()
        if num_classes is not None:
            config_.update({'TARGET': {'NUM_CLASSES': num_classes}})

        self.config = config_.make_config(config_file, validate_config, **kwargs)
        self._make_modules()

    def forward(self, image_batch, annotations_batch=None):
        if self.training:
            assert annotations_batch is not None

        features = self.fpn(self.backbone(image_batch))

        anchors = self.compute_anchors(features)
        cls_output[level], reg_output[level] = compute_cls_reg_output(features)

        anchors    = self.combine_levels(anchors)
        cls_output = self.combine_levels(cls_output)
        reg_output = self.combine_levels(reg_output)

        # if self.training:
        #     self.compute_targets(annotations_batch, anchors)
        #     loss
        # else:
        #     detections

        return anchors, cls_output, reg_output

    def combine_levels(self, x_dict):
        x = [x_dict[i] for i in self.feature_levels]
        return torch.cat(x, dim=-2)

    def compute_cls_reg_output(self, features):
        cls_output = {}
        reg_output = {}

        for level in self.feature_levels:
            if self.config.COMBINED.USE:
                cls_output[level], reg_output[level] = self.detector(features[level])
            else:
                cls_output[level] = self.classifier(features[level])
                reg_output[level] = self.regressor(features[level])

        return cls_output, reg_output


    # ##########################################################################
    # _make_modules
    # ##########################################################################

    def _make_modules(self):
        self.backbone = Backbone(**self.config.BACKBONE)
        self.fpn = FPN(**self.config.FPN)

        num_anchors = len(self.config.ANCHOR.RATIOS) * self.config.ANCHOR.SCALES_PER_OCTAVE
        self.feature_levels = range(self.config.FPN.MIN_LEVEL, self.config.FPN.MAX_LEVEL + 1)

        if self.config.COMBINED.USE:
            self.detector = CombinedHead(
                input_feature_size=self.config.FPN.FEATURE_SIZE,
                feature_size=self.config.COMBINED.FEATURE_SIZE,
                num_layers=self.config.COMBINED.NUM_LAYERS,
                num_anchors=num_anchors,
                num_classes=self.config.TARGET.NUM_CLASSES,
                use_class_specific_bbox=self.config.TARGET.CLASS_SPECIFIC_BBOX,
                use_background_predictor=self.config.TARGET.BACKGROUND_PREDICTOR,
                prior_prob=self.config.INITIALIZATION.PRIOR_PROB
            )
        else:
            self.classifier = ClassificationHead(
                input_feature_size=self.config.FPN.FEATURE_SIZE,
                feature_size=self.config.CLASSIFIER.FEATURE_SIZE,
                num_layers=self.config.CLASSIFIER.NUM_LAYERS,
                num_anchors=num_anchors,
                num_classes=self.config.TARGET.NUM_CLASSES,
                use_background_predictor=self.config.TARGET.BACKGROUND_PREDICTOR,
                prior_prob=self.config.INITIALIZATION.PRIOR_PROB
            )
            self.regressor = RegressionHead(
                input_feature_size=self.config.FPN.FEATURE_SIZE,
                feature_size=self.config.REGRESSOR.FEATURE_SIZE,
                num_layers=self.config.REGRESSOR.NUM_LAYERS,
                num_anchors=num_anchors,
                num_classes=self.config.TARGET.NUM_CLASSES,
                use_class_specific_bbox=self.config.TARGET.CLASS_SPECIFIC_BBOX,
            )

        self.compute_anchors = ComputeAnchors(
            feature_levels=self.feature_levels,
            ratios=self.config.ANCHOR.RATIOS,
            scales_per_octave=self.config.ANCHOR.SCALES_PER_OCTAVE,
            size_mult=self.config.ANCHOR.SIZE_MULT,
            stride_mult=self.config.ANCHOR.STRIDE_MULT
        )

        self.compute_targets = ComputeTargets(
            num_classes=self.config.TARGET.NUM_CLASSES,
            use_class_specific_bbox=self.config.TARGET.CLASS_SPECIFIC_BBOX,
            use_background_predictor=self.config.TARGET.BACKGROUND_PREDICTOR,
            positive_overlap=self.config.TARGET.POSITIVE_OVERLAP,
            negative_overlap=self.config.TARGET.NEGATIVE_OVERLAP
        )
