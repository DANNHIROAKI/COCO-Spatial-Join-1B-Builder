#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COCO-Spatial-Join-1B builder

This script implements the COCO‑Spatial-Join‑1B Build Spec (README) provided by the user.
It:
  - parses COCO 2017 train/val JSON
  - assigns global z_idx ordering (train first, then val; coco_image_id asc within split)
  - writes images.parquet
  - generates GT boxes (including iscrowd=1) with required normalization
  - runs Detectron2 Faster R-CNN R50-FPN RPN to obtain proposals
  - outputs exactly 10,000 proposals per image with deterministic sorting
  - writes rect shards (<=1024 images/shard, no cross-split mixing)
  - writes meta/build_manifest.json and meta/stats.json
  - performs mandatory per-image & global validations on-the-fly

Run:
  python build_coco_spatial_1b.py --help

Note:
  This build produces ~1.23B proposal rows. Expect very large compute/storage needs.

"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------
# Build constants (SPEC-fixed)
# ---------------------------
BUILDER_VERSION = "coco-spatial-join-1b-builder-1.0.0"

SPLIT_TRAIN = 0  # train2017
SPLIT_VAL = 1    # val2017

RECT_GT = 0
RECT_PROPOSAL = 1

PROPOSALS_PER_IMAGE = 10_000
STRIDE = 20_000  # rect_id stride per image (SPEC-fixed)
SHARD_IMAGE_STRIDE = 1024  # max images per shard (SPEC-fixed)

# RPN test-time params (SPEC-fixed)
RPN_PRE_NMS_TOPK_TEST = 20_000
RPN_POST_NMS_TOPK_TEST = 20_000
RPN_NMS_THRESH = 1.0

# ResizeShortestEdge params (SPEC-fixed)
RESIZE_SHORT_EDGE = 800
RESIZE_MAX_SIZE = 1333

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def get_cpu_model() -> str:
    # Best-effort on Linux; fall back to platform.processor()
    try:
        if Path("/proc/cpuinfo").exists():
            txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"model name\s*:\s*(.+)", txt)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def get_gpu_visible() -> Tuple[str, int]:
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = []
            for i in range(n):
                names.append(torch.cuda.get_device_name(i))
            # Combine if multiple
            uniq = sorted(set(names))
            name = "; ".join(uniq) if uniq else "cuda"
            return name, n
    except Exception:
        pass
    return "none", 0


def try_run(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
        return out.strip()
    except Exception:
        return ""


def detectron2_version_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import detectron2  # type: ignore
        info["version"] = getattr(detectron2, "__version__", "unknown")
        # best-effort git commit
        try:
            from detectron2.utils.collect_env import collect_env_info  # type: ignore
            info["collect_env_info"] = str(collect_env_info())
        except Exception:
            pass
    except Exception:
        info["version"] = "not_installed"
    return info


def torch_version_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cudnn_version"] = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
    except Exception:
        info["torch_version"] = "not_installed"
    return info


def set_global_determinism(seed_root: int) -> Dict[str, Any]:
    """Apply deterministic flags as required by the spec. Returns a dict describing the flags."""
    import torch

    # Recommended by PyTorch for deterministic cublas; harmless if unused.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed_root)
    np.random.seed(seed_root % (2**32))
    torch.manual_seed(seed_root)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_root)

    torch.backends.cudnn.benchmark = False
    # Extra: make cudnn deterministic where applicable
    torch.backends.cudnn.deterministic = True

    # Spec-required:
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    flags = {
        "seed_root": seed_root,
        "python_random_seed": seed_root,
        "numpy_seed": int(seed_root % (2**32)),
        "torch_seed": seed_root,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "use_deterministic_algorithms": True,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    return flags


def set_per_image_seeds(seed_root: int, z_idx: int) -> int:
    """Derive per-image seed (recorded in manifest), and set Python/NumPy/Torch seeds."""
    import torch

    # Simple deterministic derivation (documented in manifest):
    seed_img = (int(seed_root) + int(z_idx)) % (2**32)
    random.seed(seed_img)
    np.random.seed(seed_img)
    torch.manual_seed(seed_img)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_img)
    return int(seed_img)

@dataclass(frozen=True)
class ImageRec:
    split: int          # 0=train, 1=val
    coco_image_id: int  # COCO image id
    file_name: str
    width: int
    height: int
    z_idx: int          # global z index


def load_coco_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_image_list(train_json: Dict[str, Any], val_json: Dict[str, Any]) -> List[ImageRec]:
    # 4.1: sort key: (split, coco_image_id)
    train_imgs = [
        (int(img["id"]), str(img["file_name"]), int(img["width"]), int(img["height"]))
        for img in train_json["images"]
    ]
    val_imgs = [
        (int(img["id"]), str(img["file_name"]), int(img["width"]), int(img["height"]))
        for img in val_json["images"]
    ]
    train_imgs.sort(key=lambda x: x[0])
    val_imgs.sort(key=lambda x: x[0])

    out: List[ImageRec] = []
    z = 0
    for coco_id, fn, w, h in train_imgs:
        out.append(ImageRec(split=SPLIT_TRAIN, coco_image_id=coco_id, file_name=fn, width=w, height=h, z_idx=z))
        z += 1
    for coco_id, fn, w, h in val_imgs:
        out.append(ImageRec(split=SPLIT_VAL, coco_image_id=coco_id, file_name=fn, width=w, height=h, z_idx=z))
        z += 1
    return out


def group_annotations_by_image(coco_json: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    # image_id -> list[ann]
    m: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_json["annotations"]:
        img_id = int(ann["image_id"])
        m.setdefault(img_id, []).append(ann)
    return m

def _sanitize_and_clip(v: np.ndarray, upper: np.float32) -> np.ndarray:
    """Apply SPEC 6.1 non-finite mapping then clip to [0, upper], all in float32."""
    v = v.astype(np.float32, copy=False)
    # non-finite handling (before clip)
    v = np.nan_to_num(v, nan=np.float32(0.0), posinf=upper, neginf=np.float32(0.0)).astype(np.float32, copy=False)
    # clip
    v = np.clip(v, np.float32(0.0), upper).astype(np.float32, copy=False)
    return v


def _fix_degenerate(min_v: np.ndarray, max_v: np.ndarray, upper: np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SPEC 6.2 degeneracy fix in float32."""
    min_v = min_v.astype(np.float32, copy=False)
    max_v = max_v.astype(np.float32, copy=False)

    bad = max_v <= min_v
    if np.any(bad):
        max_v = max_v.copy()
        min_v = min_v.copy()

        max_v[bad] = np.nextafter(min_v[bad], np.float32(np.inf)).astype(np.float32, copy=False)

        overflow = bad & (max_v > upper)
        if np.any(overflow):
            max_v[overflow] = upper
            min_v[overflow] = np.nextafter(upper, np.float32(-np.inf)).astype(np.float32, copy=False)

    return min_v, max_v


def normalize_boxes_xyxy(
    boxes_xyxy: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize boxes to satisfy SPEC 6. Returns (min_x,min_y,max_x,max_y) float32."""
    assert boxes_xyxy.ndim == 2 and boxes_xyxy.shape[1] == 4
    w_up = np.float32(float(width))
    h_up = np.float32(float(height))

    x1 = _sanitize_and_clip(boxes_xyxy[:, 0], w_up)
    y1 = _sanitize_and_clip(boxes_xyxy[:, 1], h_up)
    x2 = _sanitize_and_clip(boxes_xyxy[:, 2], w_up)
    y2 = _sanitize_and_clip(boxes_xyxy[:, 3], h_up)

    x1, x2 = _fix_degenerate(x1, x2, w_up)
    y1, y2 = _fix_degenerate(y1, y2, h_up)

    # Final assertions (SPEC 6)
    if not (np.all(np.isfinite(x1)) and np.all(np.isfinite(y1)) and np.all(np.isfinite(x2)) and np.all(np.isfinite(y2))):
        raise ValueError("Non-finite coordinates after normalization")
    if not (np.all(x1 >= 0) and np.all(x2 <= w_up) and np.all(y1 >= 0) and np.all(y2 <= h_up)):
        raise ValueError("Out-of-bounds coordinates after normalization")
    if not (np.all(x1 < x2) and np.all(y1 < y2)):
        raise ValueError("Degenerate boxes after normalization")

    return x1, y1, x2, y2

def sigmoid_float32(logits: np.ndarray) -> np.ndarray:
    """Stable sigmoid, output float32 in [0,1]."""
    x = logits.astype(np.float64, copy=False)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    out = out.astype(np.float32)
    # Ensure finite and within [0,1]
    out = np.nan_to_num(out, nan=np.float32(0.0), posinf=np.float32(1.0), neginf=np.float32(0.0)).astype(np.float32, copy=False)
    out = np.clip(out, np.float32(0.0), np.float32(1.0)).astype(np.float32, copy=False)
    return out

def images_schema() -> pa.Schema:
    return pa.schema([
        ("split", pa.int8()),
        ("coco_image_id", pa.int32()),
        ("file_name", pa.string()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("z_idx", pa.int32()),
    ])


def rects_schema() -> pa.Schema:
    return pa.schema([
        ("rect_id", pa.int64()),
        ("z_min", pa.int32()),
        ("z_max", pa.int32()),
        ("min_x", pa.float32()),
        ("min_y", pa.float32()),
        ("max_x", pa.float32()),
        ("max_y", pa.float32()),
        ("type", pa.int8()),
        ("rank", pa.int16()),
        ("score", pa.float32()),
        ("category_id", pa.int16()),
        ("iscrowd", pa.int8()),
        ("coco_ann_id", pa.int64()),
        ("coco_image_id", pa.int32()),
    ])


def write_images_parquet(images: List[ImageRec], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split = np.array([img.split for img in images], dtype=np.int8)
    coco_image_id = np.array([img.coco_image_id for img in images], dtype=np.int32)
    file_name = [img.file_name for img in images]
    width = np.array([img.width for img in images], dtype=np.int32)
    height = np.array([img.height for img in images], dtype=np.int32)
    z_idx = np.array([img.z_idx for img in images], dtype=np.int32)

    # Ensure sorted by z_idx (SPEC 8.3)
    if not np.all(z_idx[:-1] < z_idx[1:]):
        raise ValueError("images list not sorted by z_idx")

    table = pa.Table.from_arrays(
        [
            pa.array(split, type=pa.int8()),
            pa.array(coco_image_id, type=pa.int32()),
            pa.array(file_name, type=pa.string()),
            pa.array(width, type=pa.int32()),
            pa.array(height, type=pa.int32()),
            pa.array(z_idx, type=pa.int32()),
        ],
        schema=images_schema(),
    )
    pq.write_table(table, out_path, compression="zstd")

class RectBatchAccumulator:
    def __init__(self, schema: pa.Schema):
        self.schema = schema
        self.cols: Dict[str, List[np.ndarray]] = {name: [] for name in schema.names}
        self.num_rows: int = 0

    def append(self, rows: Dict[str, np.ndarray]) -> None:
        n = None
        for k in self.schema.names:
            arr = rows[k]
            if n is None:
                n = int(arr.shape[0])
            else:
                if int(arr.shape[0]) != n:
                    raise ValueError(f"Column length mismatch for {k}: {arr.shape[0]} vs {n}")
            self.cols[k].append(arr)
        self.num_rows += int(n or 0)

    def flush_to_writer(self, writer: pq.ParquetWriter) -> int:
        if self.num_rows == 0:
            return 0

        arrays: List[pa.Array] = []
        for field in self.schema:
            name = field.name
            np_list = self.cols[name]
            col = np.concatenate(np_list, axis=0)
            arrays.append(pa.array(col, type=field.type))
        table = pa.Table.from_arrays(arrays, schema=self.schema)

        writer.write_table(table)
        written = self.num_rows

        # Reset
        self.cols = {name: [] for name in self.schema.names}
        self.num_rows = 0
        return written

def score_bin_edges() -> List[float]:
    # [0.000, 0.001, ..., 1.000] (1001 edges)
    return [round(i / 1000.0, 3) for i in range(0, 1001)]


def area_bin_edges() -> List[int]:
    # [0, 1, 2, 4, 8, ..., 2^24]
    edges = [0]
    for k in range(0, 25):  # 2^0 .. 2^24
        edges.append(2 ** k)
    return edges


def aspect_ratio_bin_edges() -> List[float]:
    # [0] + [2^k for k=-8..8] + [2^9]
    edges: List[float] = [0.0]
    for k in range(-8, 9):  # -8..8
        edges.append(float(2.0 ** k))
    edges.append(float(2.0 ** 9))
    return edges


def update_score_hist(score_counts: np.ndarray, scores_f32: np.ndarray) -> None:
    # 1000 bins: idx = floor(score * 1000), clamp 999 for score==1.0
    idx = (scores_f32.astype(np.float64) * 1000.0).astype(np.int64)
    idx = np.clip(idx, 0, 999)
    score_counts += np.bincount(idx, minlength=1000).astype(np.int64)


def update_area_hist(
    area_counts: np.ndarray,
    overflow_counter: List[int],
    min_x: np.ndarray,
    min_y: np.ndarray,
    max_x: np.ndarray,
    max_y: np.ndarray,
    edges: np.ndarray,
) -> None:
    # area = (max_x-min_x)*(max_y-min_y) in float64
    w = (max_x.astype(np.float64) - min_x.astype(np.float64))
    h = (max_y.astype(np.float64) - min_y.astype(np.float64))
    area = w * h
    # bins for [edges[i], edges[i+1]), overflow >= edges[-1]
    idx = np.searchsorted(edges, area, side="right") - 1  # -1..len(edges)-1
    overflow = idx >= (len(edges) - 1)
    if np.any(overflow):
        overflow_counter[0] += int(np.sum(overflow))
    valid = (~overflow) & (idx >= 0)
    if np.any(valid):
        area_counts += np.bincount(idx[valid], minlength=(len(edges) - 1)).astype(np.int64)


def update_ar_hist(
    ar_counts: np.ndarray,
    underflow_counter: List[int],
    overflow_counter: List[int],
    min_x: np.ndarray,
    min_y: np.ndarray,
    max_x: np.ndarray,
    max_y: np.ndarray,
    edges: np.ndarray,
) -> None:
    # ar = (w/h) in float64
    w = (max_x.astype(np.float64) - min_x.astype(np.float64))
    h = (max_y.astype(np.float64) - min_y.astype(np.float64))
    ar = w / h

    underflow = ar < edges[1]  # < 2^-8
    overflow = ar >= edges[-1] # >= 2^9
    if np.any(underflow):
        underflow_counter[0] += int(np.sum(underflow))
    if np.any(overflow):
        overflow_counter[0] += int(np.sum(overflow))

    valid = (~underflow) & (~overflow)
    if np.any(valid):
        idx = np.searchsorted(edges, ar[valid], side="right") - 1
        # idx should be >=1 and <= len(edges)-2
        ar_counts += np.bincount(idx, minlength=(len(edges) - 1)).astype(np.int64)

def build_detectron2_model(model_config_id: str, weights_path: Path, device: str) -> Tuple[Any, Any, Any, str]:
    """Build Detectron2 model and augmentation. Returns (model, aug, cfg, checkpoint_url)."""
    import torch
    from detectron2.config import get_cfg  # type: ignore
    from detectron2 import model_zoo  # type: ignore
    from detectron2.checkpoint import DetectionCheckpointer  # type: ignore
    from detectron2.modeling import build_model  # type: ignore
    from detectron2.data import transforms as T  # type: ignore

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config_id))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = device

    # SPEC 5.5 (fixed)
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = RPN_PRE_NMS_TOPK_TEST
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = RPN_POST_NMS_TOPK_TEST
    cfg.MODEL.RPN.NMS_THRESH = RPN_NMS_THRESH

    # SPEC 5.4 (fixed)
    cfg.INPUT.MIN_SIZE_TEST = RESIZE_SHORT_EDGE
    cfg.INPUT.MAX_SIZE_TEST = RESIZE_MAX_SIZE

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # Deterministic, fixed ResizeShortestEdge (DefaultPredictor-style)
    aug = T.ResizeShortestEdge([RESIZE_SHORT_EDGE, RESIZE_SHORT_EDGE], RESIZE_MAX_SIZE)

    ckpt_url = model_zoo.get_checkpoint_url(model_config_id)
    return model, aug, cfg, ckpt_url


def get_rpn_candidates(
    model: Any,
    aug: Any,
    image_bgr_u8: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run RPN and return (boxes_xyxy_in_orig, objectness_logits, cand_pool_size)."""
    import torch

    orig_h, orig_w = image_bgr_u8.shape[:2]

    tfm = aug.get_transform(image_bgr_u8)
    image_resized = tfm.apply_image(image_bgr_u8)

    # HWC BGR uint8 -> CHW float32
    image_tensor = torch.as_tensor(image_resized.astype("float32").transpose(2, 0, 1))
    image_tensor = image_tensor.to(device)

    inputs = [{"image": image_tensor, "height": orig_h, "width": orig_w}]

    with torch.inference_mode():
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)

    inst = proposals[0]
    boxes = inst.proposal_boxes.tensor.detach().cpu().numpy()  # (N,4) in resized coords
    logits = inst.objectness_logits.detach().cpu().numpy()     # (N,)

    # Map back to original image coords (SPEC 5.4)
    boxes_orig = tfm.inverse().apply_box(boxes)

    return boxes_orig, logits, int(boxes_orig.shape[0])

def build_gt_rows(img: ImageRec, anns: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    # Sort by coco_ann_id asc (SPEC 10.3)
    anns_sorted = sorted(anns, key=lambda a: int(a["id"]))
    num_gt = len(anns_sorted)
    if num_gt > PROPOSALS_PER_IMAGE:
        raise ValueError(f"SPEC violation: num_gt={num_gt} > 10000 for z_idx={img.z_idx} (coco_image_id={img.coco_image_id})")

    if num_gt == 0:
        # Return empty arrays with correct dtypes
        empty_i64 = np.zeros((0,), dtype=np.int64)
        empty_i32 = np.zeros((0,), dtype=np.int32)
        empty_f32 = np.zeros((0,), dtype=np.float32)
        empty_i16 = np.zeros((0,), dtype=np.int16)
        empty_i8 = np.zeros((0,), dtype=np.int8)
        return {
            "rect_id": empty_i64,
            "z_min": empty_i32,
            "z_max": empty_i32,
            "min_x": empty_f32,
            "min_y": empty_f32,
            "max_x": empty_f32,
            "max_y": empty_f32,
            "type": empty_i8,
            "rank": empty_i16,
            "score": empty_f32,
            "category_id": empty_i16,
            "iscrowd": empty_i8,
            "coco_ann_id": empty_i64,
            "coco_image_id": empty_i32,
        }

    bboxes = np.array([a["bbox"] for a in anns_sorted], dtype=np.float32)  # [x,y,w,h]
    x = bboxes[:, 0]
    y = bboxes[:, 1]
    w = bboxes[:, 2]
    h = bboxes[:, 3]
    boxes = np.stack([x, y, x + w, y + h], axis=1).astype(np.float32)

    min_x, min_y, max_x, max_y = normalize_boxes_xyxy(boxes, img.width, img.height)

    base = np.int64(img.z_idx) * np.int64(STRIDE)
    gt_local_idx = np.arange(num_gt, dtype=np.int64)
    rect_id = base + gt_local_idx

    z_min = np.full((num_gt,), img.z_idx, dtype=np.int32)
    z_max = np.full((num_gt,), img.z_idx + 1, dtype=np.int32)

    type_arr = np.full((num_gt,), RECT_GT, dtype=np.int8)
    rank = np.zeros((num_gt,), dtype=np.int16)
    score = np.full((num_gt,), np.float32(1.0), dtype=np.float32)

    category_id = np.array([int(a["category_id"]) for a in anns_sorted], dtype=np.int16)
    iscrowd = np.array([int(a.get("iscrowd", 0)) for a in anns_sorted], dtype=np.int8)
    coco_ann_id = np.array([int(a["id"]) for a in anns_sorted], dtype=np.int64)
    coco_image_id = np.full((num_gt,), int(img.coco_image_id), dtype=np.int32)

    return {
        "rect_id": rect_id.astype(np.int64),
        "z_min": z_min,
        "z_max": z_max,
        "min_x": min_x.astype(np.float32),
        "min_y": min_y.astype(np.float32),
        "max_x": max_x.astype(np.float32),
        "max_y": max_y.astype(np.float32),
        "type": type_arr,
        "rank": rank,
        "score": score,
        "category_id": category_id,
        "iscrowd": iscrowd,
        "coco_ann_id": coco_ann_id,
        "coco_image_id": coco_image_id,
    }


def build_proposal_rows(
    img: ImageRec,
    image_bgr_u8: np.ndarray,
    model: Any,
    aug: Any,
    device: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    boxes_orig, logits, cand_n = get_rpn_candidates(model, aug, image_bgr_u8, device)

    # SPEC 5.5: verify candidate pool >= 10000
    if cand_n < PROPOSALS_PER_IMAGE:
        raise ValueError(f"SPEC violation: candidate pool size {cand_n} < 10000 for z_idx={img.z_idx} (coco_image_id={img.coco_image_id})")

    cand_idx = np.arange(cand_n, dtype=np.int32)

    # Normalize coords BEFORE sorting (SPEC 5.6)
    min_x, min_y, max_x, max_y = normalize_boxes_xyxy(boxes_orig.astype(np.float32), img.width, img.height)

    # Score = sigmoid(objectness_logit), float32
    scores = sigmoid_float32(logits.astype(np.float32))

    # Sort key (SPEC 5.6.1):
    # 1) score desc
    # 2) (min_x,min_y,max_x,max_y) asc
    # 3) cand_idx asc
    order = np.lexsort((cand_idx, max_y, max_x, min_y, min_x, -scores))
    keep = order[:PROPOSALS_PER_IMAGE]

    min_x_k = min_x[keep].astype(np.float32, copy=False)
    min_y_k = min_y[keep].astype(np.float32, copy=False)
    max_x_k = max_x[keep].astype(np.float32, copy=False)
    max_y_k = max_y[keep].astype(np.float32, copy=False)
    scores_k = scores[keep].astype(np.float32, copy=False)
    cand_idx_k = cand_idx[keep].astype(np.int32, copy=False)

    rank = (np.arange(PROPOSALS_PER_IMAGE, dtype=np.int16) + 1)  # 1..10000

    base = np.int64(img.z_idx) * np.int64(STRIDE)
    rect_id = base + np.int64(PROPOSALS_PER_IMAGE) + (rank.astype(np.int64) - 1)

    z_min = np.full((PROPOSALS_PER_IMAGE,), img.z_idx, dtype=np.int32)
    z_max = np.full((PROPOSALS_PER_IMAGE,), img.z_idx + 1, dtype=np.int32)

    type_arr = np.full((PROPOSALS_PER_IMAGE,), RECT_PROPOSAL, dtype=np.int8)
    category_id = np.full((PROPOSALS_PER_IMAGE,), -1, dtype=np.int16)
    iscrowd = np.zeros((PROPOSALS_PER_IMAGE,), dtype=np.int8)
    coco_ann_id = np.full((PROPOSALS_PER_IMAGE,), -1, dtype=np.int64)
    coco_image_id = np.full((PROPOSALS_PER_IMAGE,), int(img.coco_image_id), dtype=np.int32)

    rows = {
        "rect_id": rect_id.astype(np.int64),
        "z_min": z_min,
        "z_max": z_max,
        "min_x": min_x_k,
        "min_y": min_y_k,
        "max_x": max_x_k,
        "max_y": max_y_k,
        "type": type_arr,
        "rank": rank,
        "score": scores_k,
        "category_id": category_id,
        "iscrowd": iscrowd,
        "coco_ann_id": coco_ann_id,
        "coco_image_id": coco_image_id,
    }
    return rows, cand_idx_k

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def validate_image_rows(img: ImageRec, gt_rows: Dict[str, np.ndarray], prop_rows: Dict[str, np.ndarray], cand_idx_k: np.ndarray) -> None:
    w_up = np.float32(float(img.width))
    h_up = np.float32(float(img.height))

    # ---- GT checks (SPEC 13.1 #5/#6/#7)
    if gt_rows["rect_id"].shape[0] > 0:
        _assert(gt_rows["rank"].dtype == np.int16, "GT rank dtype must be int16")
        _assert(np.all(gt_rows["rank"] == 0), f"GT rank must be 0 for z_idx={img.z_idx}")
        _assert(gt_rows["score"].dtype == np.float32, "GT score dtype must be float32")
        _assert(np.all(gt_rows["score"] == np.float32(1.0)), f"GT score must be 1.0 for z_idx={img.z_idx}")
        _assert(np.all(gt_rows["type"] == RECT_GT), f"GT type must be {RECT_GT} for z_idx={img.z_idx}")
        # z_max = z_min + 1
        _assert(np.all(gt_rows["z_max"] == (gt_rows["z_min"] + 1)), f"GT z_max != z_min+1 for z_idx={img.z_idx}")
        # coords finite float32
        for k in ["min_x","min_y","max_x","max_y"]:
            _assert(gt_rows[k].dtype == np.float32, f"GT {k} dtype must be float32")
            _assert(np.all(np.isfinite(gt_rows[k])), f"GT {k} must be finite for z_idx={img.z_idx}")
        # bounds and strict
        _assert(np.all((gt_rows["min_x"] >= 0) & (gt_rows["max_x"] <= w_up) & (gt_rows["min_x"] < gt_rows["max_x"])), f"GT x bounds violated for z_idx={img.z_idx}")
        _assert(np.all((gt_rows["min_y"] >= 0) & (gt_rows["max_y"] <= h_up) & (gt_rows["min_y"] < gt_rows["max_y"])), f"GT y bounds violated for z_idx={img.z_idx}")

    # ---- PROPOSAL checks (SPEC 13.1 #1-#4/#6/#7)
    n = prop_rows["rect_id"].shape[0]
    _assert(n == PROPOSALS_PER_IMAGE, f"PROPOSAL rows != 10000 for z_idx={img.z_idx}: {n}")

    # rank == 1..10000
    _assert(prop_rows["rank"].dtype == np.int16, "PROPOSAL rank dtype must be int16")
    _assert(np.array_equal(prop_rows["rank"], (np.arange(PROPOSALS_PER_IMAGE, dtype=np.int16) + 1)), f"PROPOSAL rank must be 1..10000 for z_idx={img.z_idx}")

    # score float32 finite
    _assert(prop_rows["score"].dtype == np.float32, "PROPOSAL score dtype must be float32")
    _assert(np.all(np.isfinite(prop_rows["score"])), f"PROPOSAL score must be finite for z_idx={img.z_idx}")

    # score non-increasing
    s = prop_rows["score"]
    _assert(np.all(s[:-1] >= s[1:]), f"PROPOSAL score not non-increasing for z_idx={img.z_idx}")

    # z_max = z_min + 1
    _assert(np.all(prop_rows["z_max"] == (prop_rows["z_min"] + 1)), f"PROPOSAL z_max != z_min+1 for z_idx={img.z_idx}")

    # coords finite float32 and in-bounds strict
    for k in ["min_x","min_y","max_x","max_y"]:
        _assert(prop_rows[k].dtype == np.float32, f"PROPOSAL {k} dtype must be float32")
        _assert(np.all(np.isfinite(prop_rows[k])), f"PROPOSAL {k} must be finite for z_idx={img.z_idx}")
    _assert(np.all((prop_rows["min_x"] >= 0) & (prop_rows["max_x"] <= w_up) & (prop_rows["min_x"] < prop_rows["max_x"])), f"PROPOSAL x bounds violated for z_idx={img.z_idx}")
    _assert(np.all((prop_rows["min_y"] >= 0) & (prop_rows["max_y"] <= h_up) & (prop_rows["min_y"] < prop_rows["max_y"])), f"PROPOSAL y bounds violated for z_idx={img.z_idx}")

    # Tie-breaker verification (SPEC 13.1 #4)
    # Check the sorted order matches lex order on:
    #   (-score, min_x, min_y, max_x, max_y, cand_idx)
    key0 = (-prop_rows["score"]).astype(np.float32, copy=False)
    key1 = prop_rows["min_x"]
    key2 = prop_rows["min_y"]
    key3 = prop_rows["max_x"]
    key4 = prop_rows["max_y"]
    key5 = cand_idx_k.astype(np.int32, copy=False)

    # Vectorized lexicographic nondecreasing check for consecutive pairs
    a0, b0 = key0[:-1], key0[1:]
    a1, b1 = key1[:-1], key1[1:]
    a2, b2 = key2[:-1], key2[1:]
    a3, b3 = key3[:-1], key3[1:]
    a4, b4 = key4[:-1], key4[1:]
    a5, b5 = key5[:-1], key5[1:]

    lt0 = a0 < b0
    eq0 = a0 == b0
    lt1 = a1 < b1
    eq1 = a1 == b1
    lt2 = a2 < b2
    eq2 = a2 == b2
    lt3 = a3 < b3
    eq3 = a3 == b3
    lt4 = a4 < b4
    eq4 = a4 == b4
    lt5 = a5 < b5
    eq5 = a5 == b5

    lex_ok = lt0 | (eq0 & (lt1 | (eq1 & (lt2 | (eq2 & (lt3 | (eq3 & (lt4 | (eq4 & (lt5 | eq5))))))))))
    _assert(bool(np.all(lex_ok)), f"PROPOSAL sort key order violated for z_idx={img.z_idx}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", required=True, type=str, help="COCO 2017 root dir containing train2017/, val2017/, annotations/")
    p.add_argument("--train-json", required=True, type=str, help="instances_train2017.json path")
    p.add_argument("--val-json", required=True, type=str, help="instances_val2017.json path")
    p.add_argument("--output", required=True, type=str, help="Output dir (will create coco-spatial-1b structure)")
    p.add_argument("--model-config-id", required=True, type=str, help="Detectron2 model zoo config id")
    p.add_argument("--weights", required=True, type=str, help="Local checkpoint path (.pkl) for detectron2 model")
    p.add_argument("--seed-root", required=True, type=int, help="Root seed (single int)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    p.add_argument("--batch-images", default=8, type=int, help="Flush parquet writer every N images (memory/perf tradeoff)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output dir if exists")
    p.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "none"], help="Parquet compression codec")
    p.add_argument("--limit-images", default=None, type=int, help="DEBUG: only process first K images (violates spec if set)")
    return p.parse_args()


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            # Remove only the expected output subdirs (be conservative)
            for sub in ["meta", "data"]:
                p = path / sub
                if p.exists():
                    import shutil
                    shutil.rmtree(p)
        else:
            # if non-empty, error
            if any(path.iterdir()):
                raise FileExistsError(f"Output dir exists and is not empty: {path}. Use --overwrite to overwrite.")
    path.mkdir(parents=True, exist_ok=True)

def main() -> None:
    args = parse_args()

    coco_root = Path(args.coco_root)
    train_json_path = Path(args.train_json)
    val_json_path = Path(args.val_json)
    out_root = Path(args.output)
    weights_path = Path(args.weights)

    ensure_empty_dir(out_root, overwrite=bool(args.overwrite))

    # Output structure (SPEC 7)
    meta_dir = out_root / "meta"
    data_dir = out_root / "data"
    rects_train_dir = data_dir / "rects" / "train2017"
    rects_val_dir = data_dir / "rects" / "val2017"
    meta_dir.mkdir(parents=True, exist_ok=True)
    rects_train_dir.mkdir(parents=True, exist_ok=True)
    rects_val_dir.mkdir(parents=True, exist_ok=True)

    # ----------------
    # Stage 1: parse COCO + build ImagesAll + z_idx mapping (SPEC 11.1)
    # ----------------
    print("[build] Loading COCO JSON...")
    train_json = load_coco_json(train_json_path)
    val_json = load_coco_json(val_json_path)

    images_all = build_image_list(train_json, val_json)
    num_images_total = len(images_all)
    train_count = sum(1 for im in images_all if im.split == SPLIT_TRAIN)
    val_count = sum(1 for im in images_all if im.split == SPLIT_VAL)

    print(f"[build] num_images_total={num_images_total} (train={train_count}, val={val_count})")

    # SPEC expects exact counts for COCO 2017
    # Keep as strict assertions by default.
    _assert(train_count == 118_287, f"Expected 118287 train images, got {train_count}")
    _assert(val_count == 5_000, f"Expected 5000 val images, got {val_count}")
    _assert(num_images_total == 123_287, f"Expected 123287 total images, got {num_images_total}")

    # ----------------
    # Stage 2: write images.parquet (SPEC 11.2, 9.1, 8.3)
    # ----------------
    print("[build] Writing images.parquet ...")
    images_parquet_path = data_dir / "images.parquet"
    write_images_parquet(images_all, images_parquet_path)

    # ----------------
    # Stage 3: group annotations (SPEC 11.3)
    # ----------------
    train_anns_by_img = group_annotations_by_image(train_json)
    val_anns_by_img = group_annotations_by_image(val_json)

    # ----------------
    # Stage 4: determinism settings (SPEC 5.2)
    # ----------------
    print("[build] Applying deterministic settings ...")
    import torch
    determinism_flags = set_global_determinism(int(args.seed_root))

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device=cuda but torch.cuda.is_available() is False")

    # ----------------
    # Stage 5: build model (SPEC 5.1/5.5)
    # ----------------
    print("[build] Building Detectron2 model ...")
    model, aug, cfg, checkpoint_url = build_detectron2_model(args.model_config_id, weights_path, device)

    # ----------------
    # Stage 6: prepare stats collectors (SPEC 12.2)
    # ----------------
    score_counts = np.zeros((1000,), dtype=np.int64)

    area_edges_list = area_bin_edges()
    area_edges = np.array(area_edges_list, dtype=np.float64)
    proposal_area_counts = np.zeros((len(area_edges_list) - 1,), dtype=np.int64)
    proposal_area_overflow = [0]

    ar_edges_list = aspect_ratio_bin_edges()
    ar_edges = np.array(ar_edges_list, dtype=np.float64)
    ar_counts = np.zeros((len(ar_edges_list) - 1,), dtype=np.int64)
    ar_underflow = [0]
    ar_overflow = [0]

    gt_area_counts = np.zeros((len(area_edges_list) - 1,), dtype=np.int64)
    gt_area_overflow = [0]
    total_gt = 0
    total_gt_crowd = 0

    num_gt_total_by_z: List[int] = []
    num_gt_crowd_by_z: List[int] = []

    # ----------------
    # Stage 7: write rects shards (SPEC 8, 9.2, 10, 11.3-11.5)
    # ----------------
    rect_schema = rects_schema()
    shards_manifest: List[Dict[str, Any]] = []

    def split_name(split_id: int) -> str:
        return "train2017" if split_id == SPLIT_TRAIN else "val2017"

    def split_img_dir(split_id: int) -> Path:
        return coco_root / split_name(split_id)

    def ann_map(split_id: int) -> Dict[int, List[Dict[str, Any]]]:
        return train_anns_by_img if split_id == SPLIT_TRAIN else val_anns_by_img

    total_proposals_written = 0
    total_rect_rows_written = 0

    import cv2  # type: ignore

    # Processing order must follow z_idx total order (SPEC 5.2, 8.2)
    # We process train shards first, then val shards; within each, z asc.
    split_ranges = [
        (SPLIT_TRAIN, 0, train_count, rects_train_dir),
        (SPLIT_VAL, train_count, num_images_total, rects_val_dir),
    ]

    limit_images = int(args.limit_images) if args.limit_images is not None else None
    if limit_images is not None:
        print(f"[build] WARNING: --limit-images={limit_images} set. This violates the official spec outputs.")

    global_image_counter = 0
    last_rect_id_written: Optional[int] = None

    for split_id, split_start, split_end, out_split_dir in split_ranges:
        out_split_dir.mkdir(parents=True, exist_ok=True)

        # shards start at split_start and increment by 1024 (SPEC 8.2)
        for start_z in range(split_start, split_end, SHARD_IMAGE_STRIDE):
            end_z = min(start_z + SHARD_IMAGE_STRIDE, split_end)

            # If debug limit-images cuts earlier, truncate.
            if limit_images is not None:
                if global_image_counter >= limit_images:
                    break
                # ensure we don't exceed
                remaining = limit_images - global_image_counter
                end_z = min(end_z, start_z + remaining)
                if end_z <= start_z:
                    break

            shard_file = out_split_dir / f"shard-{start_z:06d}.parquet"
            rel_shard_file = str(Path("data") / "rects" / split_name(split_id) / shard_file.name)

            print(f"[build] Writing shard: {shard_file.name}  z_idx=[{start_z},{end_z})  split={split_name(split_id)}")

            # Writer
            compression = None if args.compression == "none" else args.compression
            writer = pq.ParquetWriter(str(shard_file), rect_schema, compression=compression)

            acc = RectBatchAccumulator(rect_schema)
            shard_rows_written = 0

            # Iterate images in this shard
            for z_idx in tqdm(range(start_z, end_z), desc=f"shard {start_z:06d}", unit="img"):
                img = images_all[z_idx]
                _assert(img.z_idx == z_idx, "z_idx mismatch in images_all ordering")
                _assert(img.split == split_id, "Split mismatch while iterating shards")

                # per-image seed derivation (recorded in manifest)
                _ = set_per_image_seeds(int(args.seed_root), z_idx)

                # Load original image
                img_path = split_img_dir(split_id) / img.file_name
                im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if im is None:
                    raise FileNotFoundError(f"Failed to read image: {img_path}")
                if im.shape[0] != img.height or im.shape[1] != img.width:
                    raise ValueError(f"Image size mismatch for {img_path}: expected ({img.height},{img.width}), got ({im.shape[0]},{im.shape[1]})")

                # GT rows
                anns = ann_map(split_id).get(img.coco_image_id, [])
                gt_rows = build_gt_rows(img, anns)

                # per-image GT stats
                num_gt = int(gt_rows["rect_id"].shape[0])
                num_crowd = int(np.sum(gt_rows["iscrowd"])) if num_gt > 0 else 0
                num_gt_total_by_z.append(num_gt)
                num_gt_crowd_by_z.append(num_crowd)

                total_gt += num_gt
                total_gt_crowd += num_crowd

                # GT area hist
                if num_gt > 0:
                    update_area_hist(
                        gt_area_counts,
                        gt_area_overflow,
                        gt_rows["min_x"],
                        gt_rows["min_y"],
                        gt_rows["max_x"],
                        gt_rows["max_y"],
                        area_edges,
                    )

                # Proposal rows
                prop_rows, cand_idx_k = build_proposal_rows(img, im, model, aug, device)

                # per-image mandatory validation
                validate_image_rows(img, gt_rows, prop_rows, cand_idx_k)

                # Proposals stats
                update_score_hist(score_counts, prop_rows["score"])
                update_area_hist(
                    proposal_area_counts,
                    proposal_area_overflow,
                    prop_rows["min_x"],
                    prop_rows["min_y"],
                    prop_rows["max_x"],
                    prop_rows["max_y"],
                    area_edges,
                )
                update_ar_hist(
                    ar_counts,
                    ar_underflow,
                    ar_overflow,
                    prop_rows["min_x"],
                    prop_rows["min_y"],
                    prop_rows["max_x"],
                    prop_rows["max_y"],
                    ar_edges,
                )

                # Append to shard accumulator in rect_id order (SPEC 8.3)
                # Within an image: GT (rect_id=base+0..), then proposals (base+10000..)
                if num_gt > 0:
                    acc.append(gt_rows)
                acc.append(prop_rows)

                # rect_id monotonic check across images (SPEC 8.3)
                # Determine first rect_id written for this image:
                if num_gt > 0:
                    first_rect = int(gt_rows["rect_id"][0])
                else:
                    first_rect = int(prop_rows["rect_id"][0])
                last_rect = int(prop_rows["rect_id"][-1])

                if last_rect_id_written is not None:
                    _assert(first_rect > last_rect_id_written, f"rect_id not strictly increasing at z_idx={z_idx}")
                last_rect_id_written = last_rect

                # Flush periodically
                global_image_counter += 1
                if global_image_counter % int(args.batch_images) == 0:
                    shard_rows_written += acc.flush_to_writer(writer)

            # Final flush + close
            shard_rows_written += acc.flush_to_writer(writer)
            writer.close()

            # Record shard manifest entry
            shards_manifest.append({
                "split": split_name(split_id),
                "shard_file": rel_shard_file,
                "start_z_idx": int(start_z),
                "end_z_idx": int(end_z),
                "num_rows": int(shard_rows_written),
            })

            total_rect_rows_written += int(shard_rows_written)
            # proposals always 10k per image
            total_proposals_written += int((end_z - start_z) * PROPOSALS_PER_IMAGE)

        # end shards for split
    # end splits

    # ----------------
    # Stage 8: Global acceptance checks (SPEC 13.2)
    # ----------------
    expected_num_images = num_images_total if limit_images is None else limit_images
    expected_total_proposals = expected_num_images * PROPOSALS_PER_IMAGE

    _assert(len(num_gt_total_by_z) == expected_num_images, "per-image GT stats length mismatch")
    _assert(len(num_gt_crowd_by_z) == expected_num_images, "per-image crowd stats length mismatch")

    _assert(total_proposals_written == expected_total_proposals, f"Total proposals mismatch: {total_proposals_written} vs expected {expected_total_proposals}")

    # images.parquet row count is expected_num_images only if limit_images used, but we wrote full images.parquet.
    # For official build (limit_images None), enforce exact.
    if limit_images is None:
        _assert(expected_num_images == 123_287, "SPEC expects full 123287 images")
        _assert(expected_total_proposals == 1_232_870_000, "SPEC expects full 1,232,870,000 proposals")

    # z_idx coverage check (SPEC 13.2)
    # We assigned sequential z_idx; validate quickly.
    z_list = [im.z_idx for im in images_all[:expected_num_images]]
    _assert(z_list[0] == 0 and z_list[-1] == expected_num_images - 1, "z_idx coverage mismatch")
    _assert(len(set(z_list)) == expected_num_images, "z_idx duplicates found")

    # ----------------
    # Stage 9: Write meta files (SPEC 12)
    # ----------------
    coco_train_sha256 = sha256_file(train_json_path)
    coco_val_sha256 = sha256_file(val_json_path)
    checkpoint_sha256 = sha256_file(weights_path)

    gpu_model, gpu_count = get_gpu_visible()
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    # Build manifest
    manifest: Dict[str, Any] = {
        # COCO input
        "coco_train_json": train_json_path.name,
        "coco_val_json": val_json_path.name,
        "coco_train_json_sha256": coco_train_sha256,
        "coco_val_json_sha256": coco_val_sha256,

        # Builder env
        "builder_version": BUILDER_VERSION,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pyarrow_version": pa.__version__,
        "detectron2_version": detectron2_version_info(),
        **torch_version_info(),

        # Determinism
        "seed_root": int(args.seed_root),
        "seed_rules": "Per-image seed = (seed_root + z_idx) mod 2^32; applied to Python random, NumPy, torch (and torch.cuda if available).",
        "determinism_flags": determinism_flags,

        # Model
        "model_config_id": str(args.model_config_id),
        "rpn_pre_nms_topk_test": int(RPN_PRE_NMS_TOPK_TEST),
        "rpn_post_nms_topk_test": int(RPN_POST_NMS_TOPK_TEST),
        "rpn_nms_thresh": float(RPN_NMS_THRESH),
        "checkpoint_url": checkpoint_url,
        "checkpoint_sha256": checkpoint_sha256,

        # Build info
        "build_time_utc": utc_now_iso(),
        "host_os": platform.platform(),
        "cpu_model": get_cpu_model(),
        "gpu_model": gpu_model,
        "gpu_count": int(gpu_count),
        "ram_gb": ram_gb,

        # Output summary
        "num_images_total": int(expected_num_images if limit_images is None else expected_num_images),
        "num_proposals_total": int(expected_total_proposals),
        "shards": shards_manifest,
    }

    (meta_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Stats.json
    gt_iscrowd_ratio = (float(total_gt_crowd) / float(total_gt)) if total_gt > 0 else 0.0
    stats: Dict[str, Any] = {
        "per_image": {
            "num_gt_total": num_gt_total_by_z,
            "num_gt_crowd": num_gt_crowd_by_z,
        },
        "proposals": {
            "score_histogram": {
                "bin_edges": score_bin_edges(),
                "counts": score_counts.tolist(),
            },
            "area_histogram": {
                "bin_edges": area_edges_list,
                "counts": proposal_area_counts.tolist(),
                "overflow_count_ge_2p24": int(proposal_area_overflow[0]),
            },
            "aspect_ratio_histogram": {
                "bin_edges": ar_edges_list,
                "counts": ar_counts.tolist(),
                "underflow_count_lt_2m8": int(ar_underflow[0]),
                "overflow_count_ge_2p9": int(ar_overflow[0]),
            },
        },
        "gt": {
            "area_histogram": {
                "bin_edges": area_edges_list,
                "counts": gt_area_counts.tolist(),
                "overflow_count_ge_2p24": int(gt_area_overflow[0]),
            },
            "gt_iscrowd_ratio": gt_iscrowd_ratio,
            "total_gt": int(total_gt),
            "total_gt_crowd": int(total_gt_crowd),
        },
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[build] SUCCESS")
    print(f"[build] images.parquet: {images_parquet_path}")
    print(f"[build] rects/train2017 shards: {rects_train_dir}")
    print(f"[build] rects/val2017 shards: {rects_val_dir}")
    print(f"[build] meta/build_manifest.json: {meta_dir/'build_manifest.json'}")
    print(f"[build] meta/stats.json: {meta_dir/'stats.json'}")


if __name__ == "__main__":
    main()
