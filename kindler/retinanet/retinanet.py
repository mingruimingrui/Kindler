import torch

from .config import config, validate_config
from ..backbone import Backbone
from ..fpn import FPN
from ._retinanet import (
    ClassificationHead,
    RegressionHead,
    CombinedHead,
    ComputeAnchors,
    ComputeTargets,
    ComputeLosses,
    FilterDetections
)


class RetinaNet(torch.nn.Module):
    """
    Retinanet from FAIR https://arxiv.org/abs/1708.02002
    """
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

        # Generate anchors at each feature level
        anchors = self.compute_anchors(features)
        anchors = self.combine_levels(anchors)

        # Generate classification and regression outputs at each feature level
        cls_output, reg_output = self.compute_cls_reg_output(features)
        cls_output = self.combine_levels(cls_output)
        reg_output = self.combine_levels(reg_output)

        if self.training:
            # Compute targets and generate loss_dict if training
            with torch.no_grad():
                cls_target, reg_target, anchor_states = self.compute_targets(annotations_batch, anchors)

            loss_dict = self.compute_losses(
                cls_output=cls_output,
                reg_output=reg_output,
                cls_target=cls_target,
                reg_target=reg_target,
                anchor_states=anchor_states
            )
            return loss_dict

        else:
            # Generate detections if evaluating
            detections = self.filter_detections(cls_output, reg_output, anchors)
            return detections

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
                use_bg_predictor=self.config.TARGET.BG_PREDICTOR,
                prior_prob=self.config.INITIALIZATION.PRIOR_PROB
            )
        else:
            self.classifier = ClassificationHead(
                input_feature_size=self.config.FPN.FEATURE_SIZE,
                feature_size=self.config.CLASSIFIER.FEATURE_SIZE,
                num_layers=self.config.CLASSIFIER.NUM_LAYERS,
                num_anchors=num_anchors,
                num_classes=self.config.TARGET.NUM_CLASSES,
                use_bg_predictor=self.config.TARGET.BG_PREDICTOR,
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
            positive_overlap=self.config.TARGET.POSITIVE_OVERLAP,
            negative_overlap=self.config.TARGET.NEGATIVE_OVERLAP
        )

        self.compute_losses = ComputeLosses(
            use_focal_loss=self.config.LOSS.USE_FOCAL,
            focal_alpha=self.config.LOSS.FOCAL_ALPHA,
            focal_gamma=self.config.LOSS.FOCAL_GAMMA,
            reg_weight=self.config.LOSS.REG_WEIGHT,
            reg_beta=self.config.LOSS.REG_BETA,
            use_bg_predictor=self.config.TARGET.BG_PREDICTOR
        )

        self.filter_detections = FilterDetections(
            apply_nms=self.config.EVAL.APPLY_NMS,
            class_specific_nms=self.config.EVAL.CLASS_SPECIFIC_NMS,
            pre_nms_top_n=self.config.EVAL.PRE_NMS_TOP_N,
            post_nms_top_n=self.config.EVAL.POST_NMS_TOP_N,
            nms_thresh=self.config.EVAL.NMS_THRESH,
            score_thresh=self.config.EVAL.SCORE_THRESH,
            bg_thresh=self.config.EVAL.BG_THRESH,
            use_bg_predictor=self.config.TARGET.BG_PREDICTOR
        )
