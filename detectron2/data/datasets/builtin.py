# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os
import glob
import random

from detectron2.data import MetadataCatalog, DatasetCatalog
from .register_coco import register_coco_instances, register_coco_panoptic_separated
from .lvis import register_lvis_instances, get_lvis_instances_meta
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .pascal_voc import register_pascal_voc
from .builtin_meta import _get_builtin_metadata


# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014_100.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/train2017", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/val2017", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/val2017", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/test2017", "lvis/lvis_v0.5_image_info_test.json"),
    }
}


def register_all_lvis(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========


_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val", "cityscapes/gtFine/val"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test", "cityscapes/gtFine/test"),
}


def register_all_cityscapes(root="datasets"):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="sem_seg", **meta
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root="datasets"):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# Register them all under "./datasets"
register_all_coco()
register_all_lvis()
register_all_cityscapes()
register_all_pascal_voc()

# # register custom datasets
# register_coco_instances(
#     "xview_instance_segmentation_dataset_train", 
#     {}, 
#     "./datasets/xview_instance_segmentation_dataset_train.json", 
#     "./data/original_train/images")
# register_coco_instances(
#     "xview_instance_segmentation_dataset_val", 
#     {}, 
#     "./datasets/xview_instance_segmentation_dataset_val.json", 
#     "./data/original_train/images")
# register_coco_instances(
#     "combined_xview_instance_segmentation_dataset_train", 
#     {}, 
#     "./datasets/combined_xview_instance_segmentation_dataset_train.json", 
#     "./data/train/images")
# register_coco_instances(
#     "combined_xview_instance_segmentation_dataset_val", 
#     {}, 
#     "./datasets/combined_xview_instance_segmentation_dataset_val.json", 
#     "./data/train/images")
# register_coco_instances(
#     "xview_damage_assessment_instance_segmentation_dataset_train", 
#     {}, 
#     "./datasets/xview_damage_assessment_instance_segmentation_dataset_train.json", 
#     "./data/original_train/images")
# register_coco_instances(
#     "xview_damage_assessment_instance_segmentation_dataset_val", 
#     {}, 
#     "./datasets/xview_damage_assessment_instance_segmentation_dataset_val.json", 
#     "./data/original_train/images")
# register_coco_instances(
#     "combined_xview_damage_assessment_instance_segmentation_dataset_train", 
#     {}, 
#     "./datasets/combined_xview_damage_assessment_instance_segmentation_dataset_train.json", 
#     "./data/train/images")
# register_coco_instances(
#     "combined_xview_damage_assessment_instance_segmentation_dataset_val", 
#     {}, 
#     "./datasets/combined_xview_damage_assessment_instance_segmentation_dataset_val.json", 
#     "./data/train/images")
# register_coco_instances(
#     "inria_buildings_annotations", 
#     {}, 
#     "./datasets/inria_buildings_annotations.json", 
#     "./data/inria/train/images")

# http://code.activestate.com/recipes/303060-group-a-list-into-sequential-n-tuples/
def group(lst, n):
    """group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]
    
    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.
    
    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    return zip(*[lst[i::n] for i in range(n)])


SPLIT_PERCENT = 0.97
IMAGE_PATH = "./data/train/images/"
IMAGE_PATH_GT = "./data/train_gt/"

# each file corresponds to an image
annotation_files = sorted(glob.glob("{}/*".format(IMAGE_PATH)))
annotation_files_as_pairs = list(group(annotation_files, 2))
random.Random(4).shuffle(annotation_files_as_pairs)
index = int(SPLIT_PERCENT*len(annotation_files_as_pairs))
# now create the lists
pre_train_files = []
pre_train_files_gt = []

pre_val_files = []
pre_val_files_gt = []

post_train_files = []
post_train_files_gt = []

post_val_files = []
post_val_files_gt = []

for (post, pre) in annotation_files_as_pairs[0:index]:
    pre_train_files.append(pre)
    pre_train_files_gt.append(pre.replace(IMAGE_PATH, IMAGE_PATH_GT))
    post_train_files.append(post)
    post_train_files_gt.append(post.replace(IMAGE_PATH, IMAGE_PATH_GT))
for (post, pre) in annotation_files_as_pairs[index:]:
    pre_val_files.append(pre)
    pre_val_files_gt.append(pre.replace(IMAGE_PATH, IMAGE_PATH_GT))
    post_val_files.append(post)
    post_val_files_gt.append(post.replace(IMAGE_PATH, IMAGE_PATH_GT))

# loc_filenames = sorted(glob.glob("./data/train/images/*post*"))
# loc_filenames_gt = sorted(glob.glob("./data/train_gt/*post*"))
# assert len(loc_filenames) > 0
# assert len(loc_filenames) == len(loc_filenames_gt)
# random.Random(4).shuffle(loc_filenames)
# random.Random(4).shuffle(loc_filenames_gt)
# loc_split_idx = int(0.90*len(loc_filenames))

# Register for semantic segmentation.
def get_dicts(pre_or_post, train_or_test, input_folder=None):
    def return_func():
        dataset_dicts = []
        if train_or_test == "train":
            if pre_or_post == "pre":
                train_filenames = pre_train_files
                segmentation_filenames = pre_train_files_gt
            else:
                train_filenames = post_train_files
                segmentation_filenames = post_train_files_gt
        else:
            if pre_or_post == "pre":
                train_filenames = pre_val_files
                segmentation_filenames = pre_val_files_gt
            else:
                train_filenames = post_val_files
                segmentation_filenames = post_val_files_gt
        image_id = 0
        for train_filename, segmentation_filename in zip(train_filenames, segmentation_filenames):
            record = {}
            record["file_name"] = train_filename
            if input_folder:
                record["file_name"] = train_filename.replace(IMAGE_PATH, input_folder)
            record["height"] = 1024
            record["width"] = 1024
            record["image_id"] = image_id
            record["sem_seg_file_name"] = segmentation_filename
            image_id += 1
            dataset_dicts.append(record)
        return dataset_dicts
    return return_func

# standard
DatasetCatalog.register("xview_semantic_localization_train", get_dicts("pre", "train"))
DatasetCatalog.register("xview_semantic_localization_val", get_dicts("pre", "val"))
DatasetCatalog.register("xview_semantic_damage_train", get_dicts("post", "train"))
DatasetCatalog.register("xview_semantic_damage_val", get_dicts("post", "val"))

# merged color channels
DatasetCatalog.register(
    "xview_semantic_damage_pre_post_dark_train",
    get_dicts("post", "train", input_folder="./data/train_pre_post/")
)
DatasetCatalog.register(
    "xview_semantic_damage_pre_post_dark_val",
    get_dicts("post", "val", input_folder="./data/train_pre_post/")
)
