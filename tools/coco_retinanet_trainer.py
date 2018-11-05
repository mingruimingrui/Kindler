import os
import sys

# Append Kindler root directory to sys
test_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(test_dir, os.pardir))
sys.path.append(root_dir)

import json
import yaml
import logging
import argparse

import torch
from torch.utils.data import ConcatDataset, DataLoader

# Trainer
from kindler.engine import do_train

# Model and optimizer
from kindler.retinanet import RetinaNet
from kindler.solver import make_sgd_optimizer, WarmupMultiStepLR

# Dataset
from kindler.data.datasets import CocoDataset
from kindler.data.samplers import DetectionSampler
from kindler.data.collate import ImageCollate
from kindler.data import transforms

# Env and logging
from kindler.utils.logger import setup_logging
from kindler.utils.comm import is_main_process
from kindler.utils.collect_env import collect_env_info

logger = logging.getLogger(__name__)


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def parse_args(args):
    parser = argparse.ArgumentParser('Trainer for detection model')

    parser.add_argument('-y', '--yaml-file', type=str,
        help='Path to yaml file containing script configs')

    parser.add_argument('--model-config-file', type=str,
        help='Path to model config file')
    parser.add_argument('--ann-files', type=str, nargs='+',
        help='Path to annotation files multiple files can be accepted')
    parser.add_argument('--root-img-dirs', type=str, nargs='+',
        help='Path to image directories, should have same entries as ann files')

    parser.add_argument('--batch-size', type=int, default=1,
        help='Size of batches during training')
    parser.add_argument('--max-iter', type=int, default=1440000,
        help='Maximum number of iterations to perform during training')
    parser.add_argument('--base-lr', type=float, default=0.001,
        help='Learning rate to use during training')
    parser.add_argument('--warmup-iters', type=int, default=8000,
        help='Number of iterations for SGD warm up')

    parser.add_argument('--log-dir', type=str, default='./',
        help='Directory to store log files')
    parser.add_argument('--checkpoint-dir', type=str, default='./',
        help='Directory to store checkpoint files')

    parser.add_argument('--local_rank', type=int, default=0,
        help='For torch.distributed.launch')

    return parser.parse_args(args)


def config_args(args):
    """
    Does a number of things
    - Ensure that args are valid
    - Create directory and files
    - Set up logging
    """
    if args.yaml_file is not None:
        with open(args.yaml_file, 'r') as f:
            yaml_configs = yaml.load(f)
        for key, value in yaml_configs.items():
            assert hasattr(args, key), \
                '{} is an invalid option'.format(key)
            setattr(args, key, value)

    if isinstance(args.ann_files, str):
        args.ann_files = [args.ann_files]
    if isinstance(args.root_img_dirs, str):
        args.root_img_dirs = [args.root_img_dirs]

    assert len(args.ann_files) == len(args.root_img_dirs)

    assert args.batch_size > 0
    assert args.max_iter > 0
    assert args.base_lr > 0

    makedirs(args.log_dir)
    makedirs(args.checkpoint_dir)

    return args


def make_lr_scheduler(optimizer, args):
    """
    Make learning rate scheduler as recommended in
    https://arxiv.org/abs/1706.02677
    """
    milestones = [0, int(args.max_iter * 2 / 3), int(args.max_iter * 8 / 9)]
    return WarmupMultiStepLR(
        optimizer,
        milestones=milestones,
        warmup_iters=args.warmup_iters
    )


def make_data_loader(args):
    image_transforms = transforms.Compose([
        # transforms.ImageResize(min_size=800, max_size=1333),
        transforms.ImageResize(min_size=800, max_size=1333),
        transforms.RandomHorizontalFlip(),
        transforms.ImageNormalization(),
        transforms.ToTensor()
    ])
    image_collate = ImageCollate()

    datasets = []
    for root_img_dir, ann_file in zip(args.root_img_dirs, args.ann_files):
        datasets.append(CocoDataset(
            root_img_dir,
            ann_file,
            mask=False,
            transforms=image_transforms
        ))

    coco_dataset = ConcatDataset(datasets)
    batch_sampler = DetectionSampler(
        coco_dataset,
        batch_size=args.batch_size,
        random_sample=True,
        num_iter=args.max_iter
    )

    return DataLoader(
        coco_dataset,
        collate_fn=image_collate,
        batch_sampler=batch_sampler,
        num_workers=args.batch_size * 2,
        # pin_memory=True
    )


def loss_fn(model, batch):
    """ loss_fn as required by do_train """
    return model(batch['image'], batch['annotations'])


def main(args):
    torch.cuda.set_device(args.local_rank)

    # Identify if using distributed training
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        torch.distributed.deprecated.init_process_group(
            backend='nccl',
            init_method='env://'
        )

    # Make model
    model = RetinaNet(config_file=args.model_config_file).cuda()
    model_config = model.config
    assert model.config.TARGET.NUM_CLASSES == 80, \
        'Requires RetinaNet config file to define num_classes as 80'

    # Make optimizer
    optimizer = make_sgd_optimizer(model, args.base_lr)
    scheduler = make_lr_scheduler(optimizer, args)

    # Make model distributed
    if distributed:
        model = torch.nn.parallel.deprecated.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False
        )

    # Make data loader
    data_loader = make_data_loader(args)

    # Log only if main process
    if is_main_process():
        setup_logging(os.path.join(args.log_dir, 'train.log'))

    logger.info('Collecting env info')
    logger.info('\n' + collect_env_info() + '\n')

    logger.info('Using {} GPUs\n'.format(num_gpus))

    logger.info('Training config: {}\n'.format(
        json.dumps(vars(args), indent=2)
    ))

    logger.info('Model config: {}\n'.format(
        json.dumps(dict(model_config), indent=2)
    ))

    do_train(
        model=model,
        data_loader=data_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    args = config_args(args)
    main(args)
