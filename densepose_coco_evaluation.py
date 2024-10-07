# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE file in the root directory of this source tree.
# This is a modified version of cocoeval.py where we also have the densepose evaluation.

import os
import copy
import h5py
import pickle
import datetime
import numpy as np
from enum import Enum
from scipy.io import loadmat
from collections import defaultdict
import scipy.spatial.distance as ssd
from scipy.ndimage import zoom as spzoom
from pycocotools import mask as mask_utils


NUM_I_PARTS = 24
MEAN_GEODESIC_DISTANCES = [0, 0.351, 0.107, 0.126, 0.237, 0.173, 0.142, 0.128, 0.150]
COARSE_PARTS_LABELS = [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8]


class DensePoseDataRelative:
    """
    Dense pose relative annotations that can be applied to any bounding box:
        x - normalized X coordinates [0, 255] of annotated points
        y - normalized Y coordinates [0, 255] of annotated points
        i - body part labels 0,...,24 for annotated points
        u - body part U coordinates [0, 1] for annotated points
        v - body part V coordinates [0, 1] for annotated points
        segm - 256x256 segmentation mask with values 0,...,14
    To obtain absolute x and y data wrt some bounding box one needs to first
    divide the data by 256, multiply by the respective bounding box size
    and add bounding box offset:
        x_img = x0 + x_norm * w / 256.0
        y_img = y0 + y_norm * h / 256.0
    Segmentation masks are typically sampled to get image-based masks.
    """

    # Key for normalized X coordinates in annotation dict
    X_KEY = "dp_x"
    # Key for normalized Y coordinates in annotation dict
    Y_KEY = "dp_y"
    # Key for U part coordinates in annotation dict (used in chart-based annotations)
    U_KEY = "dp_U"
    # Key for V part coordinates in annotation dict (used in chart-based annotations)
    V_KEY = "dp_V"
    # Key for I point labels in annotation dict (used in chart-based annotations)
    I_KEY = "dp_I"
    # Key for segmentation mask in annotation dict
    S_KEY = "dp_masks"


class DensePoseEvalMode(str, Enum):
    # use both masks and geodesic distances (GPS * IOU) to compute scores
    GPSM = "gpsm"
    # use only geodesic distances (GPS)  to compute scores
    GPS = "gps"
    # use only masks (IOU) to compute scores
    IOU = "iou"


class Params:
    """
    Params for coco evaluation api
    """
    def set_uv_params(self):
        self.imgIds = []
        self.catIds = []
        self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0**2, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
        self.areaRngLbl = ["all", "medium", "large"]

    def __init__(self):
        self.set_uv_params()


def get_densepose_mask(polys):
    mask_gen = np.zeros([256, 256])
    stop = min(len(polys) + 1, 15)
    for i in range(1, stop):
        if polys[i - 1]:
            current_mask = mask_utils.decode(polys[i - 1])
            mask_gen[current_mask > 0] = i
    return mask_gen


def extract_iuv_from_quantized(dt, py, px, pt_mask):
    dt = dt["densepose"].labels_uv_uint8.numpy()
    ipoints = dt[0, py, px]
    upoints = dt[1, py, px] / 255.0  # convert from uint8 by /255.
    vpoints = dt[2, py, px] / 255.0
    ipoints[pt_mask == -1] = 0
    return ipoints, upoints, vpoints


def get_distances_uv(c_verts_gt, c_verts_det, p_dist_matrix):
    n = 27554
    dists = []
    for d in range(len(c_verts_gt)):
        if c_verts_gt[d] > 0:
            if c_verts_det[d] > 0:
                i = c_verts_gt[d] - 1
                j = c_verts_det[d] - 1
                if j == i:
                    dists.append(0)
                else:
                    if j > i:
                        i, j = j, i
                    i = n - i - 1
                    j = n - j - 1
                    k = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1
                    k = (n * n - n) / 2 - k - 1
                    dists.append(p_dist_matrix[int(k)][0])
            else:
                dists.append(np.inf)
    return np.atleast_1d(np.array(dists).squeeze())


class DensePoseCocoEval:
    """
    Interface for evaluating detection on the Microsoft COCO dataset.

    The usage for CocoEval is as follows:
     cocoGt=..., cocoDt=...       # load dataset and results
     E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
     E.params.recThrs = ...;      # set parameters as desired
     E.evaluate();                # run per image evaluation
     E.accumulate();              # accumulate per image results
     E.summarize();               # display summary metrics of results
    For example usage see evalDemo.m and http://mscoco.org/.

    The evaluation parameters are as follows (defaults in brackets):
     imgIds     - [all] N img ids to use for evaluation
     catIds     - [all] K cat ids to use for evaluation
     iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
     recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
     areaRng    - [...] A=4 object area ranges for evaluation
     maxDets    - [1 10 100] M=3 thresholds on max detections per image
    Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.

    evaluate(): evaluates detections on every image and every category and
    concats the results into the "evalImgs" with fields:
     dtIds      - [1xD] id for each of the D detections (dt)
     gtIds      - [1xG] id for each of the G ground truths (gt)
     dtMatches  - [TxD] matching gt id at each IoU or 0
     gtMatches  - [TxG] matching dt id at each IoU or 0
     dtScores   - [1xD] confidence of each dt
     gtIgnore   - [1xG] ignore flag for each gt
     dtIgnore   - [TxD] ignore flag for each dt at each IoU

    accumulate(): accumulates the per-image, per-category evaluation
    results in "evalImgs" into the dictionary "eval" with fields:
     params     - parameters used for evaluation
     date       - date evaluation was performed
     counts     - [T,R,K,A,M] parameter dimensions (see above)
     precision  - [TxRxKxAxM] precision for every evaluation setting
     recall     - [TxKxAxM] max recall for every evaluation setting
    Note: precision and recall==-1 for settings with no gt objects.

    See also coco, mask, pycocoDemo, pycocoEvalDemo

    Microsoft COCO Toolbox.      version 2.0
    Data, paper, and tutorials available at:  http://mscoco.org/
    Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    Licensed under the Simplified BSD License [see coco/license.txt]
    """
    def __init__(
        self,
        cocoGt=None,
        cocoDt=None,
        dpEvalMode: DensePoseEvalMode = DensePoseEvalMode.GPS,
    ):
        """
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        """
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self._dpEvalMode = dpEvalMode
        self.evalImgs = defaultdict(list)  # per-image per-category eval results [KxAxI]
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params()  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        self.params.imgIds = sorted(cocoGt.getImgIds())
        self.params.catIds = sorted(cocoGt.getCatIds())

    def _load_g_eval(self):
        smpl_subdiv_fpath = "./DensePoseData/eval_data/SMPL_subdiv.mat"
        pdist_transform_fpath = "./DensePoseData/eval_data/SMPL_SUBDIV_TRANSFORM.mat"
        pdist_matrix_fpath = "./DensePoseData/eval_data/Pdist_matrix.mat"

        dirname = os.path.dirname(__file__)
        smpl_subdiv = loadmat(os.path.join(dirname, smpl_subdiv_fpath))
        self.pdist_transform = loadmat(os.path.join(dirname, pdist_transform_fpath))
        self.pdist_transform = self.pdist_transform["index"].squeeze().astype(np.uint32)
        uv = np.array([smpl_subdiv["U_subdiv"], smpl_subdiv["V_subdiv"]]).squeeze()
        part_id_subdiv = smpl_subdiv["Part_ID_subdiv"].squeeze()
        closest_vert_inds = np.arange(uv.shape[1]) + 1
        self.part_uvs = []
        self.part_closest_vert_inds = []
        for i in np.arange(NUM_I_PARTS):
            self.part_uvs.append(uv[:, part_id_subdiv == (i + 1)])
            self.part_closest_vert_inds.append(closest_vert_inds[part_id_subdiv == (i + 1)])

        with h5py.File(os.path.join(dirname, pdist_matrix_fpath), 'r') as arrays:
            self.pdist_matrix = arrays['Pdist_matrix'][()]
        self.part_ids = np.array(part_id_subdiv)
        self.mean_distances = np.array(MEAN_GEODESIC_DISTANCES)
        self.coarse_parts = np.array(COARSE_PARTS_LABELS)

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        """

        p = self.params

        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))

        imns = self.cocoGt.loadImgs(p.imgIds)
        self.size_mapping = {im['id']: [im["height"], im["width"]] for im in imns}

        self._load_g_eval()

        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            gt["ignore"] = ("dp_x" in gt) == 0

        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)

        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        """
        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        cat_ids = p.catIds

        if self._dpEvalMode in {DensePoseEvalMode.GPSM, DensePoseEvalMode.IOU}:
            self.real_ious = {
                (imgId, catId): self.compute_dp_iou(imgId, catId)
                for imgId in p.imgIds
                for catId in cat_ids
            }

        self.ious = {
            (imgId, catId): self.compute_o_gps(imgId, catId) for imgId in p.imgIds for catId in cat_ids
        }

        self.evalImgs = [
            self.evaluate_img(imgId, catId, areaRng, p.maxDets[-1])
            for catId in cat_ids
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)

    def _generate_rlemask_on_image(self, mask, imgId, data):
        x, y, w, h = np.array(data["bbox"])
        im_h, im_w = self.size_mapping[imgId]
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        if mask is not None:
            x0 = max(int(x), 0)
            x1 = min(int(x + w), im_w, int(x) + mask.shape[1])
            y0 = max(int(y), 0)
            y1 = min(int(y + h), im_h, int(y) + mask.shape[0])
            y = int(y)
            x = int(x)
            im_mask[y0:y1, x0:x1] = mask[y0 - y:y1 - y, x0 - x:x1 - x]
        im_mask = np.require(np.asarray(im_mask > 0), dtype=np.uint8, requirements=["F"])
        rle_mask = mask_utils.encode(np.array(im_mask[:, :, np.newaxis], order="F"))[0]
        return rle_mask

    @staticmethod
    def layer_dts(dt, maxdets):
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > maxdets[-1]:
            return dt[0:maxdets[-1]]
        else:
            return dt

    def compute_dp_iou(self, imgId, catId):
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]

        if len(gt) == 0 and len(dt) == 0:
            return []

        dt = self.layer_dts(dt, p.maxDets)

        gtmasks = []
        for g in gt:
            if DensePoseDataRelative.S_KEY in g:
                # convert DensePose mask to a binary mask
                mask = np.minimum(get_densepose_mask(g[DensePoseDataRelative.S_KEY]), 1.0)
                _, _, w, h = g["bbox"]
                scale_x = float(max(w, 1)) / mask.shape[1]
                scale_y = float(max(h, 1)) / mask.shape[0]
                mask = spzoom(mask, (scale_y, scale_x), order=1, prefilter=False)
                mask = np.array(mask > 0.5, dtype=np.uint8)
                rle_mask = self._generate_rlemask_on_image(mask, imgId, g)
            elif "segmentation" in g:
                segmentation = g["segmentation"]
                if isinstance(segmentation, list) and segmentation:
                    # polygons
                    im_h, im_w = self.size_mapping[imgId]
                    rles = mask_utils.frPyObjects(segmentation, im_h, im_w)
                    rle_mask = mask_utils.merge(rles)
                elif isinstance(segmentation, dict):
                    if isinstance(segmentation["counts"], list):
                        # uncompressed RLE
                        im_h, im_w = self.size_mapping[imgId]
                        rle_mask = mask_utils.frPyObjects(segmentation, im_h, im_w)
                    else:
                        # compressed RLE
                        rle_mask = segmentation
                else:
                    rle_mask = self._generate_rlemask_on_image(None, imgId, g)
            else:
                rle_mask = self._generate_rlemask_on_image(None, imgId, g)
            gtmasks.append(rle_mask)

        dtmasks = []
        for d in dt:
            mask = d["densepose"].labels_uv_uint8[0].numpy()
            mask = np.require(np.asarray(mask > 0), dtype=np.uint8, requirements=["F"])
            rle_mask = self._generate_rlemask_on_image(mask, imgId, d)
            dtmasks.append(rle_mask)

        # compute iou between each dt and gt region
        iscrowd = [int(o.get("iscrowd", 0)) for o in gt]
        ious_dp = mask_utils.iou(dtmasks, gtmasks, iscrowd)
        return ious_dp

    def compute_o_gps_single_pair(self, dt, gt, py, px, pt_mask):
        i_dt, u_dt, v_dt = extract_iuv_from_quantized(dt, py, px, pt_mask)
        i_gt, u_gt, v_gt = np.array(gt["dp_I"]), np.array(gt["dp_U"]), np.array(gt["dp_V"])

        c_verts_gt, c_verts_gt_transformed = self.find_all_closest_verts_uv(u_gt, v_gt, i_gt)
        _, c_verts_det = self.find_all_closest_verts_uv(u_dt, v_dt, i_dt)

        # Get pairwise geodesic distances between gt and estimated mesh points.
        dist = get_distances_uv(c_verts_gt_transformed,
                                c_verts_det,
                                self.pdist_matrix)
        # Compute the Ogps measure.
        # Find the mean geodesic normalization distance for
        # each GT point, based on which part it is on.
        current_mean_distances = self.mean_distances[
            self.coarse_parts[
                self.part_ids[
                    c_verts_gt[c_verts_gt > 0].astype(int) - 1]
            ]
        ]
        return dist, current_mean_distances

    def compute_o_gps(self, imgId, catId):
        p = self.params
        # dimension here should be Nxm
        g = self._gts[imgId, catId]
        d = self._dts[imgId, catId]
        d = self.layer_dts(d, p.maxDets)

        if len(g) == 0 or len(d) == 0:
            return []
        ious = np.zeros((len(d), len(g)))
        # compute opgs between each detection and ground truth object
        # sigma = self.sigma #0.255 # dist = 0.3m corresponds to ogps = 0.5
        # 1 # dist = 0.3m corresponds to ogps = 0.96
        # 1.45 # dist = 1.7m (person height) corresponds to ogps = 0.5)
        for j, gt in enumerate(g):
            if not gt["ignore"]:
                g_ = gt["bbox"]
                for i, dt in enumerate(d):

                    dy = int(dt["bbox"][3])
                    dx = int(dt["bbox"][2])
                    dp_x = np.array(gt["dp_x"]) * g_[2] / 255.0
                    dp_y = np.array(gt["dp_y"]) * g_[3] / 255.0
                    py = (dp_y + g_[1] - dt["bbox"][1]).astype(int)
                    px = (dp_x + g_[0] - dt["bbox"][0]).astype(int)

                    pts = np.zeros(len(px))
                    pts[px >= dx] = -1
                    pts[py >= dy] = -1
                    pts[px < 0] = -1
                    pts[py < 0] = -1
                    if len(pts) < 1:
                        ogps = 0.0
                    elif np.max(pts) == -1:
                        ogps = 0.0
                    else:
                        px[pts == -1] = 0
                        py[pts == -1] = 0
                        dists_between_matches, dist_norm_coeffs = self.compute_o_gps_single_pair(dt, gt, py, px, pts)

                        # Compute gps
                        ogps_values = np.exp(
                            -(dists_between_matches**2) / (2 * (dist_norm_coeffs**2))
                        )

                        ogps = np.mean(ogps_values) if len(ogps_values) > 0 else 0.0
                    ious[i, j] = ogps

        gbb = [gt["bbox"] for gt in g]
        dbb = [dt["bbox"] for dt in d]

        # compute iou between each dt and gt region
        iscrowd = [int(o.get("iscrowd", 0)) for o in g]
        ious_bb = mask_utils.iou(dbb, gbb, iscrowd)
        return ious, ious_bb

    def evaluate_img(self, img_id, cat_id, a_rng, max_det):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """

        p = self.params
        gt = self._gts[img_id, cat_id]
        dt = self._dts[img_id, cat_id]

        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g["ignore"] or (g["area"] < a_rng[0] or g["area"] > a_rng[1]):
                g["_ignore"] = True
            else:
                g["_ignore"] = False

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:max_det]]
        iscrowd = [int(o.get("iscrowd", 0)) for o in gt]
        # load computed ious
        ious = (
            self.ious[img_id, cat_id][0][:, gtind]
            if len(self.ious[img_id, cat_id]) > 0
            else self.ious[img_id, cat_id]
        )
        ioubs = (
            self.ious[img_id, cat_id][1][:, gtind]
            if len(self.ious[img_id, cat_id]) > 0
            else self.ious[img_id, cat_id]
        )
        if self._dpEvalMode in {DensePoseEvalMode.GPSM, DensePoseEvalMode.IOU}:
            ious_m = (
                self.real_ious[img_id, cat_id][:, gtind]
                if len(self.real_ious[img_id, cat_id]) > 0
                else self.real_ious[img_id, cat_id]
            )

        iou_thresh_len = len(p.iouThrs)
        gtm = np.zeros((iou_thresh_len, len(gt)))
        dtm = np.zeros((iou_thresh_len, len(dt)))
        gt_ig = np.array([g["_ignore"] for g in gt])
        dt_ig = np.zeros((iou_thresh_len, len(dt)))
        if np.all(gt_ig):
            dt_ig = np.logical_or(dt_ig, True)

        if len(ious) > 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, _g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gt_ig[m] == 0 and gt_ig[gind] == 1:
                            break

                        if self._dpEvalMode == DensePoseEvalMode.GPSM:
                            new_iou = np.sqrt(ious_m[dind, gind] * ious[dind, gind])
                        elif self._dpEvalMode == DensePoseEvalMode.IOU:
                            new_iou = ious_m[dind, gind]
                        elif self._dpEvalMode == DensePoseEvalMode.GPS:
                            new_iou = ious[dind, gind]

                        if new_iou < iou:
                            continue
                        if new_iou == 0.0:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = new_iou
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dt_ig[tind, dind] = gt_ig[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]

        if not len(ioubs) == 0:
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                if dtm[tind, dind] == 0:
                    ioub = 0.8
                    m = -1
                    for gind, _g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # continue to next gt unless better match made
                        if ioubs[dind, gind] < ioub:
                            continue
                        # if match successful and best so far, store appropriately
                        ioub = ioubs[dind, gind]
                        m = gind
                        # if match made store id of match for both dt and gt
                    if m > -1:
                        dt_ig[:, dind] = gt_ig[m]
                        if gt_ig[m]:
                            dtm[tind, dind] = gt[m]["id"]
                            gtm[tind, m] = d["id"]

        # set unmatched detections outside of area range to ignore
        a = np.array([d["area"] < a_rng[0] or d["area"] > a_rng[1] for d in dt]).reshape((1, len(dt)))
        dt_ig = np.logical_or(dt_ig, np.logical_and(dtm == 0, np.repeat(a, iou_thresh_len, 0)))

        # store results for given image and category
        return {
            "image_id": img_id,
            "category_id": cat_id,
            "aRng": a_rng,
            "maxDet": max_det,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gt_ig,
            "dtIgnore": dt_ig,
        }

    def accumulate(self):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        # allows input customized parameters
        p = self.params
        p.catIds = p.catIds
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds)
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -(np.ones((T, R, K, A, M)))  # -1 for the precision of absent categories
        recall = -(np.ones((T, K, A, M)))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")

                    dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                        except Exception:
                            pass
                        precision[t, :, k, a, m] = np.array(q)

        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
        }

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this function can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(np.abs(iouThr - p.iouThrs) < 0.001)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(np.abs(iouThr - p.iouThrs) < 0.001)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            return mean_s

        def _summarize_uvs():
            stats = [_summarize(1, maxDets=self.params.maxDets[0])]
            min_threshold = self.params.iouThrs.min()
            if min_threshold <= 0.201:
                stats += [_summarize(1, maxDets=self.params.maxDets[0], iouThr=0.2)]
            if min_threshold <= 0.301:
                stats += [_summarize(1, maxDets=self.params.maxDets[0], iouThr=0.3)]
            if min_threshold <= 0.401:
                stats += [_summarize(1, maxDets=self.params.maxDets[0], iouThr=0.4)]
            stats += [
                _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.5),
                _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.75),
                _summarize(1, maxDets=self.params.maxDets[0], areaRng="medium"),
                _summarize(1, maxDets=self.params.maxDets[0], areaRng="large"),
                _summarize(0, maxDets=self.params.maxDets[0]),
                _summarize(0, maxDets=self.params.maxDets[0], iouThr=0.5),
                _summarize(0, maxDets=self.params.maxDets[0], iouThr=0.75),
                _summarize(0, maxDets=self.params.maxDets[0], areaRng="medium"),
                _summarize(0, maxDets=self.params.maxDets[0], areaRng="large"),
            ]
            return np.array(stats)

        if not self.eval:
            raise Exception("Please run accumulate() first")
        self.stats = _summarize_uvs()

    def find_all_closest_verts_uv(self, u_points, v_points, i_points):
        closest_verts = np.ones(i_points.shape) * -1
        for i in np.arange(NUM_I_PARTS):
            if (i + 1) in i_points:
                uvs = np.array([u_points[i_points == (i + 1)],
                                v_points[i_points == (i + 1)]])
                pw_dists = ssd.cdist(self.part_uvs[i].transpose(),
                                     uvs.transpose()
                                     ).squeeze()
                closest_verts[i_points == (i + 1)] = self.part_closest_vert_inds[i][np.argmin(pw_dists, axis=0)]
        closest_verts_transformed = self.pdist_transform[closest_verts.astype(int) - 1]
        closest_verts_transformed[closest_verts < 0] = 0
        return closest_verts, closest_verts_transformed
