from __future__ import absolute_import
from __future__ import division

import math
import random

import torch
import torch.utils.data

from torch.utils.data.dataset import ConcatDataset
from ..datasets import DetectionDataset


class DetectionSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler which allow user to customize sampling of DetectionDataset
    To be used with DetectionDataset
    """
    def __init__(
        self,
        dataset,
        batch_size=1,
        group_method='ratio',
        shuffle=False,
        random_sample=False,
        num_iter=None,
        drop_uneven=True,
        drop_no_anns=True
    ):
        """
        Args:
            dataset : A dataset object of either type ConcatDataset or DetectionDataset
            group_method : Either one of 'ratio', 'random' 'none'
            shuffle : Option of shuffling the order of groups
            random_sample : Every iteration randomly samples a group according to
                group_method. Must be used in conjunction with num_iter.
                This option and shuffle are about the same but implementation
                for this is a lot cleaner. Shuffle option will be ignored.
            num_iter : Number of iteration of groups to generate
            drop_uneven : Drops uneven batches
            drop_no_anns : Drops images with no annotations
        """
        assert isinstance(dataset, (ConcatDataset, DetectionDataset))
        assert group_method in {'ratio', 'random', 'none'}
        if num_iter is not None:
            assert num_iter > 0

        all_idx = list(range(len(dataset)))
        all_aspect_ratios = self._get_aspect_ratios(dataset)

        if drop_no_anns:
            # Drop items with no annotations
            all_size_anns = self._get_size_anns(dataset)
            all_idx, all_aspect_ratios = self._filter_no_anns(
                all_idx, all_aspect_ratios, all_size_anns
            )

        if group_method == 'ratio':
            new_order = np.argsort(all_aspect_ratios)

        elif group_method == 'random':
            new_order = list(range(len(all_idx)))
            random.shuffle(new_order)

        if group_method in {'ratio', 'random'}:
            all_idx, all_aspect_ratios = self._reorder_idx_aspect_ratios(
                all_idx, all_aspect_ratios, new_order
            )

        self.all_idx = all_idx
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_sample = random_sample
        self.num_iter = num_iter
        self.drop_uneven = True

    def __iter__(self):
        groups = self.generate_groups()
        return (group for group in groups)

    def __len__(self):
        if self.num_iter is None:
            return len(self.generate_groups())
        else:
            return self.num_iter

    @staticmethod
    def _get_aspect_ratios(dataset):
        is_concat_dataset = isinstance(dataset, ConcatDataset)
        dataset_size = len(dataset)
        if is_concat_dataset:
            # Get aspect ratios from DetectionDatasets in ConcatDataset.datasets
            # Gather all into a list
            all_aspect_ratios = []
            for d in dataset.datasets:
                all_aspect_ratios += self._get_aspect_ratios(d)
        else:
            # Get list of aspect ratios if DetectoinDataset
            all_aspect_ratios = [dataset.get_item_aspect_ratio(i) for i in range(dataset_size)]
        return all_aspect_ratios

    @staticmethod
    def _get_size_anns(dataset):
        is_concat_dataset = isinstance(dataset, ConcatDataset)
        dataset_size = len(dataset)
        if is_concat_dataset:
            # Get has anns from DetectionDatasets in ConcatDataset.datasets
            # Gather all into a list
            all_size_anns = []
            for d in dataset.datasets:
                all_size_anns += self._get_size_anns(d)
        else:
            # Get list of size of anns if DetectoinDataset
            all_size_anns = [dataset.get_item_num_annotations(i) for i in range(dataset_size)]
        return all_size_anns

    @staticmethod
    def _filter_no_anns(all_idx, all_aspect_ratios, all_size_anns):
        keep_idx = []
        keep_aspect_ratios = []

        for size, idx, aspect_ratio in (all_size_anns, all_idx, all_aspect_ratios):
            if size > 0:
                keep_idx.append(idx)
                keep_aspect_ratios.append(aspect_ratio)

        return keep_idx, keep_aspect_ratios

    @staticmethod
    def _reorder_idx_aspect_ratios(all_idx, all_aspect_ratios, new_order):
        reordered_idx = []
        reordered_aspect_ratios = []

        for i in new_order:
            reordered_idx.append(all_idx[i])
            reordered_aspect_ratios.append(all_aspect_ratios[i])

        return reordered_idx, reordered_aspect_ratios

    def generate_groups(self):
        if self.random_sample:
            # Assume that user also specified num_iter
            # every group (for each iter will be randomly sampled)
            max_i0 = max(0, len(self.all_idx) - self.batch_size)
            list_i0 = [random.randint(0, max_i0) for _ in range(self.num_iter)]
            groups = [
                self.all_idx[i0:i0+self.batch_size]
                for i0 in list_i0
            ]

        else:
            # Perform normal groupping
            groups = [
                self.all_idx[i0:i0+self.batch_size]
                for i0 in range(0, len(self.all_idx), self.batch_size)
            ]

            if self.drop_uneven:
                groups = [g for g in groups if len(g) == self.batch_size]

            if self.num_iter is not None:
                mult = math.ceil(self.num_iter / len(groups))
                groups = groups * int(mult)

            if self.shuffle:
                random.shuffle(groups)

            if self.num_iter is not None:
                groups = groups[:num_iter]

        return groups
