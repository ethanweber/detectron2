import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
from skimage.draw import polygon
import cv2
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import pycocotools.mask as maskUtils
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table

from .evaluator import DatasetEvaluator

class XviewEvaluator(DatasetEvaluator):
    """
    Evaluate xview results.
    """

    def __init__(self, dataset_name):
        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        # TODO(ethan): don't make these relative
        # the directories to write to
        self._PRED_DIR = "/home/ethanweber/Documents/xview/metrics/PRED_DIR"
        self._TARG_DIR = "/home/ethanweber/Documents/xview/metrics/TARG_DIR"
        self._OUT_FP = "/home/ethanweber/Documents/xview/metrics/OUT_FP"

    def reset(self):
        # TODO(ethan): delete everything at the paths where files are written
        pass

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for index, input, output in zip(range(len(inputs)), inputs, outputs):

            # WRITE THE GROUND TRUTH IMAGE
            image_id = input["image_id"]
            img = self._coco_api.loadImgs(image_id)[0]
            ann_ids = self._coco_api.getAnnIds(imgIds=[image_id])
            anns = self._coco_api.loadAnns(ann_ids)
            file_name, height, width, this_image_id = [img[i] for i in img.keys()]
            assert(image_id == this_image_id)
            image = get_xview_gt_image(file_name, height, width, image_id, anns)
            cv2.imwrite(os.path.join(self._TARG_DIR, file_name), image)

            # WRITE THE PREDICTION IMAGE
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)

                if instances.has("pred_masks"):
                    # use RLE to encode the masks, because they are too large and takes memory
                    # since this evaluator stores outputs of the entire dataset
                    # Our model may predict bool array, but cocoapi expects uint8
                    rles = [
                        mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                        for mask in instances.pred_masks
                    ]
                    for rle in rles:
                        # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                        # json writer which always produces strings cannot serialize a bytestream
                        # unless you decode it. Thankfully, utf-8 works out (which is also what
                        # the pycocotools/_mask.pyx does).
                        rle["counts"] = rle["counts"].decode("utf-8")
                    instances.pred_masks_rle = rles
                    instances.remove("pred_masks")

                prediction["instances"] = instances_to_json(instances, input["image_id"])

            # merge instances into single image and write
            # TODO: add a threshold for the results.
            # TODO: check if output already has thresholded values. probably does?
            assert("instances" in output)
            pred_image = np.zeros((height, width), 'uint8')
            for instance in prediction["instances"]:
                rle = [instance["segmentation"]]
                # TODO: add category to mask
                mask = maskUtils.decode(rle)[:, :, 0].astype("uint8")
                pred_image += mask
            cv2.imwrite(os.path.join(self._PRED_DIR, file_name), pred_image * 255)

            # NOW RUN THE EVALUATION
            # TODO: figure out the correct return format
            # return {"segm": 0}

def instances_to_json(instances, img_id):
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks_rle")
    if has_mask:
        rles = instances.pred_masks_rle

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


# https://scikit-image.org/docs/0.7.0/auto_examples/plot_shapes.html
def get_xview_gt_image(file_name, height, width, image_id, anns, include_damage=False):
    """
    Return a ground truth image with the annotations.
    :param file_name:
    :param height:
    :param width:
    :param image_id: id from COCO dataset
    :param anns (array of object): annotations to display
    :return: image (np array) that can be written to file
    """
    
    # Create an image of the correct dimension.
    image = np.zeros((height, width), 'uint8')
    for ann in anns:
        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                for seg in ann['segmentation']:
                    # Round to integers.
                    poly = np.array(seg).reshape((int(len(seg)/2), 2)).astype(int)
                    # Initialize boolean array defining shape fill.
                    rr, cc = polygon(poly[:,1], poly[:,0], image.shape)
                    image[rr,cc] = 1
    return image