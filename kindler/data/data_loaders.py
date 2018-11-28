from torch.utils.data import ConcatDataset, DataLoader
from .datasets import CocoDataset
from .samplers import DetectionSampler
from .collate import ImageCollate
from . import transforms


def make_coco_data_loader(
    root_image_dirs,
    ann_files,
    num_iter,
    batch_size=1,
    num_workers=2,
    shuffle=False,
    mask=False,
    min_size=800,
    max_size=1333,
    random_horizontal_flip=False,
    random_vertical_flip=False,
):
    """
    Creates a coco data loader
    """
    image_transforms = transforms.Compose([
        transforms.ImageResize(min_size=min_size, max_size=max_size),
        transforms.RandomHorizontalFlip(),
        transforms.ImageNormalization(),
        transforms.ToTensor()
    ])
    image_collate = ImageCollate()

    datasets = []
    for root_image_dir, ann_file in zip(root_image_dirs, ann_files):
        datasets.append(CocoDataset(
            root_image_dir,
            ann_file,
            mask=mask,
            transforms=image_transforms
        ))

    coco_dataset = ConcatDataset(datasets)
    batch_sampler = DetectionSampler(
        coco_dataset,
        batch_size=batch_size,
        random_sample=shuffle,
        num_iter=num_iter
    )

    return DataLoader(
        coco_dataset,
        collate_fn=image_collate,
        batch_sampler=batch_sampler,
        num_workers=num_workers
    )
