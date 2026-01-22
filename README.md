# COCO‑Spatial‑Join‑1B One‑Stop Builder

This repo is an **end-to-end** builder for the dataset described in your build spec.

## Quick start

```bash
bash run.sh
```

By default, it will:

1. Create & activate an environment (conda if available, else python venv)
2. Install **PyTorch**, **torchvision**, **detectron2**
3. Download **MS COCO 2017** train/val + annotations
   - Default: download from a **Hugging Face mirror** repo (often faster than the official COCO host)
   - Fallback: can use the official COCO URLs
4. Download the Detectron2 model zoo checkpoint for:
   `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml`
5. Build the dataset into:

```
coco-spatial-1b/
  meta/
    build_manifest.json
    stats.json
  data/
    images.parquet
    rects/
      train2017/
        shard-000000.parquet
        shard-001024.parquet
        ...
      val2017/
        shard-118287.parquet
        ...
```

## Config overrides (env vars)

Common overrides:

```bash
# Put COCO somewhere else:
COCO_ROOT=/data/coco2017

# Output directory:
OUTPUT_DIR=/data/coco-spatial-1b

# Force CPU (not recommended for full build):
DEVICE=cpu TORCH_CUDA=cpu

# Force a specific CUDA wheel flavor:
TORCH_CUDA=cu121

# Pin detectron2 to a specific git commit:
DETECTRON2_GIT_REF=<commit_hash>

# Root seed:
SEED_ROOT=12345

bash run.sh

# Use official COCO host instead of Hugging Face mirror
COCO_DOWNLOAD_SOURCE=official bash run.sh

# Use a different HF mirror repo (must contain train2017.zip / val2017.zip / annotations_trainval2017.zip)
HF_COCO_REPO=pcuenq/coco-2017-mirror COCO_DOWNLOAD_SOURCE=hf bash run.sh

# Disable HF transfer accelerators (enabled by default for HF downloads)
ENABLE_HF_TRANSFER=0 ENABLE_HF_XET_HIGH_PERFORMANCE=0 bash run.sh
```

## Notes

- This build writes **~1.23B proposals**, so it requires **lots** of disk.
- If `detectron2` compilation fails, install build tools (`gcc/g++`, `cmake`) and (for CUDA) a CUDA toolkit with `nvcc`.

