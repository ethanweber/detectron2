import contextlib
import io
import glob
import logging
import os
import tempfile
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

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
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            ann_ids = self._coco_api.getAnnIds(imgIds=[image_id])
            anno = self._coco_api.loadAnns(ann_ids)
            for obj in anno:
                print(obj["bbox"])