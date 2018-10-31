"""
Script used to store RetinaNet misc
"""
import torch

conv_3x3_kwargs = {'kernel': 3, 'stride': 1, 'padding': 1}
conv_1x1_kwargs = {'kernel': 1, 'stride': 1, 'padding': 0}


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
        super(ClassificationHead, self).__init__()

        self.total_num_classes = num_classes
        if use_background_predictor:
            self.total_num_classes += 1

        # Add conv tower
        tower = []
        for i in range(num_layers):
            tower.append(torch.nn.Conv2d(
                input_feature_size if i == 0 else feature_size,
                feature_size,
                **conv_3x3_kwargs
            ))
            tower.append(torch.nn.ReLU(inplace=True))

        # Add classifier layer
        tower.append(torch.nn.Conv2d(
            input_feature_size if num_layers == 0 else feature_size,
            self.total_num_classes * num_anchors,
            **conv_1x1_kwargs
        ))
        tower.append(torch.nn.Sigmoid())

        # Initialize classification output to prior_prob
        # kernel ~ 0.0
        # bias   ~ -log((1 - prior_prob) / prior_prob)  So that output is prior_prob after sigmoid
        kernel = tower[-2].weight
        bias = tower[-2].bias
        kernel.data.fill_(0.0)
        bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))

        self.tower = torch.nn.Sequential(*tower)

    def forward(self, x):
        x = self.tower(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.total_num_classes)


class RegressionHead(torch.nn.Module):
    def __init__(
        self,
        input_feature_size,
        feature_size,
        num_layers,
        num_anchors,
        num_classes=None,
        use_class_specific_bbox=False
    ):
        super(RegressionHead, self).__init__()

        self.total_num_bbox = 4
        if use_class_specific_bbox:
            assert num_classes is not None
            self.total_num_bbox *= num_classes

        # Add conv tower
        tower = []
        for i in range(num_layers):
            tower.append(torch.nn.Conv2d(
                input_feature_size if i == 0 else feature_size,
                feature_size,
                **conv_3x3_kwargs
            ))
            tower.append(torch.nn.ReLU(inplace=True))

        # Add regression layer
        tower.append(torch.nn.Conv2d(
            input_feature_size if num_layers == 0 else feature_size,
            num_anchors * self.total_num_bbox,
            **conv_1x1_kwargs
        ))

        self.tower = torch.nn.Sequential(*tower)

    def forward(self, x):
        x = self.tower(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.total_num_bbox)


class CombinedHead(torch.nn.Module):
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
        super(CombinedHead, self).__init__()

        self.total_num_classes = num_classes
        if use_background_predictor:
            self.total_num_classes += 1

        self.total_num_bbox = 4
        if use_class_specific_bbox:
            assert num_classes is not None
            self.total_num_bbox *= num_classes

        self.output_feature_size = self.total_num_classes + self.total_num_bbox

        # Add conv tower
        tower = []
        for i in range(num_layers):
            tower.append(torch.nn.Conv2d(
                input_feature_size if i == 0 else feature_size,
                feature_size,
                **conv_3x3_kwargs
            ))
            tower.append(torch.nn.ReLU(inplace=True))

        # Add combined layer
        tower.append(torch.nn.Conv2d(
            input_feature_size if num_layers == 0 else feature_size,
            num_anchors * (self.output_feature_size)
        ))

        # Initialize classification output to prior_prob
        # kernel ~ 0.0
        # bias   ~ -log((1 - prior_prob) / prior_prob)  So that output is prior_prob after sigmoid
        kernel = tower[-1].weight
        bias = tower[-1].bias
        import pdb; pdb.set_trace()
        kernel.data.fill_(0.0)
        bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))

        self.tower = torch.nn.Sequential(*tower)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.tower(x)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.output_feature_size)

        classification = self.sigmoid(x[..., :self.total_num_classes])
        regression = x[..., -self.total_num_bbox:]

        return classification, regression


class ComputeAnchors(torch.nn.Module):
    def __init__(
        self,
        feature_levels,
        ratios,
        scales_per_octave,
        size_mult,
        stride_mult,
    ):
        super(GenerateAnchors, self).__init__()
        pass
