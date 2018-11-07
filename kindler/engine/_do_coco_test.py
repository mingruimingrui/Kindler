import os
import json
import logging

import time
import datetime

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

from ..data.datasets import CocoDataset
from ..data.collate import ImageCollate
from ..data import transforms

from ..utils.misc import to_device
from ..utils import vis

logger = logging.getLogger(__name__)


def load_ann_file(ann_file):
    with open(ann_file, 'r') as f:
        ann_data = json.load(f)
    return ann_data


def load_dataset(ann_file, root_img_dir):
    image_transforms = transforms.Compose([
        transforms.ImageResize(min_size=800, max_size=1333),
        transforms.ImageNormalization(),
        transforms.ToTensor()
    ])

    # dataset_no_trans = CocoDataset(
    #     root_img_dir,
    #     ann_file,
    #     mask=False
    # )

    dataset = CocoDataset(
        root_img_dir,
        ann_file,
        mask=False,
        transforms=image_transforms
    )

    return dataset


def make_data_loader(dataset):
    return DataLoader(
        dataset,
        collate_fn=ImageCollate(),
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )


def do_coco_test(
    model,
    ann_file,
    root_img_dir,
    max_vis=100,
    vis_dir=None,
    test_type='bbox'
):
    """
    Args:
        model: The model to test
        data_loader: An torch.utils.data.DataLoader object
        loss_fn: A wrapper that takes in a model and a batch
            and outputs a training loss_dict.
            This loss_dict can be must be a dict containing a key 'total_loss'
        max_vis: TODO
        vis_dir: TODO
    """
    coco_ann_data = load_ann_file(ann_file)
    coco_dataset = load_dataset(ann_file, root_img_dir)
    data_loader = make_data_loader(coco_dataset)

    # model.train()
    model.eval()
    device = next(model.parameters()).device

    logger.info('Start coco test')
    start_testing_time = time.time()

    # for batch in data_loader:
    #     batch = to_device(batch, device)
    #
    #     loss_dict = model(batch['image'], batch['annotations'])
    #
    #     import pdb; pdb.set_trace()

    with torch.no_grad():
        for batch in data_loader:
            batch_image_size = batch['image'].shape[-2:]
            detections = model(batch['image'].cuda(non_blocking=True))

            # Something
            detections = detections[0]
            image_info = coco_dataset.coco.imgs[batch['coco_idx'][0]]

            boxes  = detections['boxes'].cpu().data.numpy()
            scores = detections['scores'].cpu().data.numpy()
            labels = detections['labels'].cpu().data.numpy()

            boxes[:, 0] = boxes[:, 0] * image_info['width'] / batch_image_size[-2]
            boxes[:, 1] = boxes[:, 1] * image_info['height'] / batch_image_size[-1]
            boxes[:, 2] = boxes[:, 2] * image_info['width'] / batch_image_size[-2]
            boxes[:, 3] = boxes[:, 3] * image_info['height'] / batch_image_size[-1]
            labels = [coco_dataset.contiguous_to_coco_cat[l] for l in labels]

            if True:
                file_path = os.path.join(root_img_dir, image_info['file_name'])
                vis_path = os.path.join(vis_dir, image_info['file_name'])

                image = Image.open(file_path)
                image = np.array(image).copy()

                vis.draw_detections(image, boxes, scores, labels)

                Image.fromarray(image.astype('uint8')).save(vis_path)

            import pdb; pdb.set_trace()

    'todo'
