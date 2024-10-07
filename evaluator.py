# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# NOTICE file in the root directory of this source tree.
# This is a modified version of evaluator.py.

import torch
import numpy as np
from dataclasses import dataclass
from densepose_coco_evaluation import DensePoseCocoEval, DensePoseEvalMode


@dataclass
class DensePoseChartResultQuantized:
    labels_uv_uint8: torch.Tensor


def _evaluate_predictions_on_coco(
    coco_gt,
    coco_results,
    multi_storage=None,
    embedder=None,
    class_names=None,
    min_threshold: float = 0.5,
    img_ids=None,
):

    densepose_metrics = _get_densepose_metrics(min_threshold)
    if len(coco_results) == 0:  # cocoapi does not handle empty results very well
        raise ValueError("coco_results is Empty")

    coco_dt = coco_gt.loadRes(coco_results)

    results = []
    for eval_mode_name in ["GPS", "GPSM", "IOU"]:
        eval_mode = getattr(DensePoseEvalMode, eval_mode_name)
        coco_eval = DensePoseCocoEval(coco_gt, coco_dt, dpEvalMode=eval_mode)
        result = _derive_results_from_coco_eval(
            coco_eval, densepose_metrics, class_names, min_threshold, img_ids
        )
        results.append(result)
    return results


def _get_densepose_metrics(min_threshold: float = 0.5):
    metrics = ["AP"]
    if min_threshold <= 0.201:
        metrics += ["AP20"]
    if min_threshold <= 0.301:
        metrics += ["AP30"]
    if min_threshold <= 0.401:
        metrics += ["AP40"]
    metrics.extend(["AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"])
    return metrics


def _derive_results_from_coco_eval(
    coco_eval, metrics, class_names, min_threshold: float, img_ids
):
    if img_ids is not None:
        coco_eval.params.imgIds = img_ids
    coco_eval.params.iouThrs = np.linspace(
        min_threshold, 0.95, int(np.round((0.95 - min_threshold) / 0.05)) + 1, endpoint=True
    )
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    results = {metric: float(coco_eval.stats[idx] * 100) for idx, metric in enumerate(metrics)}

    if class_names is None or len(class_names) <= 1:
        return results
