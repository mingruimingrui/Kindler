"""
Dataset for general detection based tasks
also covers segmentation and keypoint
"""

from __future__ import absolute_import
from __future__ import division

from pycocotools.coco import COCO

from .detection_dataset import DetectionDataset


class CocoDataset(DetectionDataset):
    def __init__(self, root_image_dir, ann_file, mask=False, transforms=None):
        self.coco = COCO(ann_file)
        self.root_dir = root_image_dir

        self._load_coco_data(mask)
        self._check_inputs()
        if mask:
            self._convert_segms_to_rles()

    def _load_coco_data(self, mask):
        def coco_idx_to_image_file(idx):
            return self.coco.imgs[idx]['file_name']

        def coco_ids_to_image_height_width(idx):
            image_info = self.coco.imgs[idx]
            return image_info['height'], image_info['width']

        def coco_idx_to_annotations(idx):
            annotation_infos = self.coco.imgToAnns[idx]
            anns = []
            for ann_info in annotation_infos:
                bbox = ann_info['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                cat_id = ann_info['category_id']
                cat_id = self.coco_cat_to_contiguous[cat_id]
                anns.append(bbox + [cat_id])
            return anns

        def coco_idx_to_segms(idx):
            annotation_infos = self.coco.imgToAnns[idx]
            return [ann['segmentation'] for ann in annotation_infos]

        coco_cat_ids = self.coco.cats.keys()
        self.coco_cat_to_contiguous = {coco_i:cont_i for cont_i, coco_i in enumerate(coco_cat_ids)}
        self.contiguous_to_coco_cat = {cont_i:coco_i for cont_i, coco_i in enumerate(coco_cat_ids)}

        self.coco_idx = self.coco.getImgIds()
        self.image_files = [coco_idx_to_image_file(idx) for idx in self.coco_idx]
        image_sizes = [coco_ids_to_image_height_width(idx) for idx in self.coco_idx]
        self.image_heights, self.image_widths = zip(*image_sizes)
        self.image_heights = list(self.image_heights)
        self.image_widths = list(self.image_widths)
        self.annotations = [coco_idx_to_annotations(idx) for idx in self.coco_idx]

        if mask:
            self.segms = [coco_idx_to_segms(idx) for idx in self.coco_idx]
