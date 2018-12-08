import torch
import torchvision

from .config import config, validate_config


class Backbone(torch.nn.Module):
    def __init__(self, config_file=None, **kwargs):
        """
        Creates a backbone model
        """
        super(Backbone, self).__init__()
        self.config = config.make_config(config_file, validate_config, **kwargs)

        if 'resnet' in self.config.TYPE:
            self._make_resnet_modules()

        if self.config.FREEZE_AT > 0:
            self._freeze_backbone()

        if self.config.FREEZE_BN:
            self._freeze_batchnorm()

        if self.config.USE_GN:
            self._use_groupnorm()

    def forward(self, x):
        features = dict()
        for i, layer_names in enumerate(self._layer_names_list, 1):
            for layer_name in layer_names:
                x = getattr(self, layer_name)(x)
            if i >= 2:
                features[i] = x
        return features

    def _make_resnet_modules(self):
        resnet_model_fn = getattr(torchvision.models, self.config.TYPE)
        resnet_model = resnet_model_fn(pretrained=self.config.PRETRAINED)
        self._layer_names_list = []

        # C1 layer
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self._layer_names_list.append([
            'conv1',
            'bn1',
            'relu'
        ])

        # C2 layer
        if self.config.LAST_CONV >= 2:
            self.maxpool = resnet_model.maxpool
            self.layer1 = resnet_model.layer1
            self._layer_names_list.append([
                'maxpool',
                'layer1'
            ])

        # C3 layer
        if self.config.LAST_CONV >= 3:
            self.layer2 = resnet_model.layer2
            self._layer_names_list.append(['layer2'])

        # C4 layer
        if self.config.LAST_CONV >= 4:
            self.layer3 = resnet_model.layer3
            self._layer_names_list.append(['layer3'])

        # C5 layer
        if self.config.LAST_CONV >= 5:
            self.layer4 = resnet_model.layer4
            self._layer_names_list.append(['layer4'])

        del resnet_model

    def _freeze_backbone(self):
        for i, layer_names in enumerate(self._layer_names_list, 1):
            if self.config.FREEZE_AT < i:
                continue
            for layer_name in layer_names:
                for param in getattr(self, layer_name).parameters():
                    param.requires_grad = False

    def _freeze_batchnorm(self):
        for layer in self.modules():
            if not isinstance(layer, torch.nn.BatchNorm2d):
                continue
            for param in layer.parameters():
                param.requires_grad = False

    def _use_groupnorm(self):
        """
        Swap all batch norm layers for group norm layers
        Group norm layers will not be frozen
        """
        errmsg = 'Channel cannot be divided by group number at {} ' + \
            'Got channel_size of {} and num_group of {}'

        for name, layer in self.named_modules():
            if not isinstance(layer, torch.nn.BatchNorm2d):
                continue

            channel_size = len(layer.weight)
            assert channel_size % self.config.GN_NUM_GROUPS == 0, \
                errmsg.format(name, channel_size, self.config.GN_NUM_GROUPS)

            path = [(int(n) if n.isdigit() else n) for n in name.split('.')]
            layer_addr = self
            for p in path:
                if isinstance(p, int):
                    layer_addr = layer_addr[p]
                elif isinstance(p, str):
                    layer_addr = getattr(layer_addr, p)

            layer_addr = torch.nn.GroupNorm(self.config.GN_NUM_GROUPS, channel_size)
