# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry

from ..backbone import build_backbone
from ..postprocessing import sem_seg_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["SemanticSegmentor", "SEM_SEG_HEADS_REGISTRY", "SemSegFPNHead", "build_sem_seg_head"]


SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")
"""
Registry for semantic segmentation heads, which make semantic segmentation predictions
from feature maps.
"""


@META_ARCH_REGISTRY.register()
class SemanticSegmentor(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            sem_seg: semantic segmentation ground truth
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "sem_seg" whose value is a
                Tensor of the output resolution that represents the
                per-pixel segmentation prediction.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        pre_images = [x["pre_image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        pre_images = [self.normalizer(x) for x in pre_images]
        # print(images[0].shape)
        # images = [torch.cat((x, y), 0) for x, y in zip(images, pre_images)]
        # print(images[0].shape)
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        pre_images = ImageList.from_tensors(pre_images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)
        pre_features = self.backbone(pre_images.tensor)
        for key in features.keys():
            # print("before")
            # print(features[key].shape)
            features[key] = torch.cat((features[key], pre_features[key]), 1)
            # print(features[key].shape)
        # print(type(features))
        # print(features.keys())
        # print(features["p2"].shape)

        if "sem_seg" in batched_inputs[0]:
            # TODO(ethan): here!
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor

            targets_localization = [x["sem_seg_localization"].to(self.device) for x in batched_inputs]
            targets_localization = ImageList.from_tensors(
                targets_localization, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            targets = None
            targets_localization = None
        results, losses = self.sem_seg_head(features, targets, targets_localization)

        if self.training:
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results


def build_sem_seg_head(cfg, input_shape):
    """
    Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)


@SEM_SEG_HEADS_REGISTRY.register()
class SemSegFPNHead(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value     = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes           = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims             = 512 # cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm                  = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight      = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            # print(self.common_stride)
            # raise ValueError("stop")
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    conv_dims, # feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features, targets=None, targets_localization=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        x = self.predictor(x)
        x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)

        if self.training:
            losses = {}
            values_1_to_4 = x[:,1:,:,:]
            values_1_to_4_sum = values_1_to_4.sum(1, keepdim=True)
            values_0 = x[:,:1,:,:]
            buildings = torch.cat((values_0, values_1_to_4_sum), 1)
            # print(buildings.shape)
            # print(values_1_to_4.shape)
            # print(values_1_to_4.sum(1, keepdim=True).shape)
            # print(targets)
            # print(targets.shape)
            losses["loss_sem_seg"] = (
                F.cross_entropy(x, targets, reduction="mean", ignore_index=self.ignore_value)
                * self.loss_weight
            )
            losses["loss_sem_seg_localization"] = (
                F.cross_entropy(buildings, targets_localization, reduction="mean", ignore_index=self.ignore_value)
                * self.loss_weight
            )
            return [], losses
        else:
            return x, {}
