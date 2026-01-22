#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# COCO-Spatial-Join-1B one-stop runner
#
# Usage:
#   bash run.sh
#
# (Optional) override via env vars:
#   USE_CONDA=auto|1|0          (default: auto)
#   ENV_NAME=coco_spatial_1b    (conda env name)
#   PYTHON_BIN=python3          (only for venv mode)
#   TORCH_CUDA=cu121|cu118|cpu  (default: auto-detect)
#   TORCH_VERSION=2.4.1
#   TORCHVISION_VERSION=0.19.1
#   DETECTRON2_GIT_REF=main     (or a commit hash)
#   SEED_ROOT=12345
#   DEVICE=auto|cuda|cpu
#   COCO_ROOT=/path/to/coco2017 (default: ./_coco2017)
#   OUTPUT_DIR=/path/to/out     (default: ./coco-spatial-1b)
#   BATCH_IMAGES=8              (flush parquet every N images)
#
#   COCO_DOWNLOAD_SOURCE=hf|official|auto  (default: hf)
#   HF_COCO_REPO=pcuenq/coco-2017-mirror   (HF dataset repo that mirrors raw COCO zips)
#   HF_COCO_REVISION=main
#   ENABLE_HF_TRANSFER=1|0                 (default: 1 when COCO_DOWNLOAD_SOURCE=hf)
#   ENABLE_HF_XET_HIGH_PERFORMANCE=1|0     (default: 1 when COCO_DOWNLOAD_SOURCE=hf)
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

USE_CONDA="${USE_CONDA:-auto}"
ENV_NAME="${ENV_NAME:-coco_spatial_1b}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SEED_ROOT="${SEED_ROOT:-12345}"
DEVICE="${DEVICE:-auto}"
BATCH_IMAGES="${BATCH_IMAGES:-8}"

COCO_ROOT="${COCO_ROOT:-$ROOT_DIR/_coco2017}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/coco-spatial-1b}"
MODEL_CONFIG_ID="${MODEL_CONFIG_ID:-COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml}"

COCO_DOWNLOAD_SOURCE="${COCO_DOWNLOAD_SOURCE:-hf}"
HF_COCO_REPO="${HF_COCO_REPO:-pcuenq/coco-2017-mirror}"
HF_COCO_REVISION="${HF_COCO_REVISION:-main}"
ENABLE_HF_TRANSFER="${ENABLE_HF_TRANSFER:-1}"
ENABLE_HF_XET_HIGH_PERFORMANCE="${ENABLE_HF_XET_HIGH_PERFORMANCE:-1}"

mkdir -p "$ROOT_DIR/_logs"
mkdir -p "$ROOT_DIR/_checkpoints"

echo "[run.sh] ROOT_DIR=$ROOT_DIR"
echo "[run.sh] COCO_ROOT=$COCO_ROOT"
echo "[run.sh] OUTPUT_DIR=$OUTPUT_DIR"
echo "[run.sh] SEED_ROOT=$SEED_ROOT"
echo "[run.sh] DEVICE=$DEVICE"
echo "[run.sh] BATCH_IMAGES=$BATCH_IMAGES"
echo "[run.sh] MODEL_CONFIG_ID=$MODEL_CONFIG_ID"
echo "[run.sh] COCO_DOWNLOAD_SOURCE=$COCO_DOWNLOAD_SOURCE"
echo "[run.sh] HF_COCO_REPO=$HF_COCO_REPO"
echo "[run.sh] HF_COCO_REVISION=$HF_COCO_REVISION"

# -----------------------------
# 1) Create/activate environment
# -----------------------------
if [[ "$USE_CONDA" == "auto" ]]; then
  if command -v conda >/dev/null 2>&1; then
    USE_CONDA="1"
  else
    USE_CONDA="0"
  fi
fi

if [[ "$USE_CONDA" == "1" ]]; then
  echo "[run.sh] Using conda environment: $ENV_NAME"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[run.sh] Creating conda env '$ENV_NAME' (python=3.10)"
    conda create -y -n "$ENV_NAME" python=3.10
  fi
  conda activate "$ENV_NAME"
else
  echo "[run.sh] Using python venv at ./.venv (system python: $PYTHON_BIN)"
  VENV_DIR="$ROOT_DIR/.venv"
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel

# -----------------------------
# 2) Install PyTorch + torchvision (if missing)
#
# IMPORTANT: With `set -e`, a failing command would abort the script.
# Bash exempts commands used as the condition of `if`, so we structure
# checks this way to allow auto-install.
# -----------------------------
if python - <<'PY'
import importlib, sys
ok = True
for m in ["torch", "torchvision"]:
    try:
        importlib.import_module(m)
    except Exception:
        ok = False
if ok:
    import torch, torchvision
    print(f"[run.sh] torch already installed: {torch.__version__}")
    print(f"[run.sh] torchvision already installed: {torchvision.__version__}")
    sys.exit(0)
print("[run.sh] torch/torchvision missing -> will install")
sys.exit(1)
PY
then
  :
else
  bash scripts/install_torch.sh
fi

# -----------------------------
# 3) Install detectron2 (if missing)
# -----------------------------
if python - <<'PY'
import importlib, sys
try:
    importlib.import_module("detectron2")
    import detectron2
    print(f"[run.sh] detectron2 already installed: {getattr(detectron2, '__version__', 'unknown')}")
    sys.exit(0)
except Exception:
    print("[run.sh] detectron2 missing -> will install")
    sys.exit(1)
PY
then
  :
else
  bash scripts/install_detectron2.sh
fi

# -----------------------------
# 4) Install remaining requirements
# -----------------------------
python -m pip install -r requirements.txt

# -----------------------------
# 5) Download COCO 2017 (images + annotations)
# -----------------------------
export COCO_DOWNLOAD_SOURCE
export HF_COCO_REPO
export HF_COCO_REVISION

# Enable Hugging Face download accelerators by default when using HF.
if [[ "$COCO_DOWNLOAD_SOURCE" == "hf" || "$COCO_DOWNLOAD_SOURCE" == "auto" ]]; then
  if [[ -z "${HF_HOME:-}" ]]; then
    export HF_HOME="$COCO_ROOT/_hf_home"
  fi

  if [[ "$ENABLE_HF_TRANSFER" == "1" ]]; then
    # Enables Rust-based hf_transfer fast-path (if installed). See HF docs.
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  fi

  if [[ "$ENABLE_HF_XET_HIGH_PERFORMANCE" == "1" ]]; then
    # Speeds up hf_xet downloads for repos stored with Xet storage.
    export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
  fi

  echo "[run.sh] HF_HOME=${HF_HOME}"
  echo "[run.sh] HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}"
  echo "[run.sh] HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-0}"
fi

bash scripts/download_coco2017.sh "$COCO_ROOT"

# -----------------------------
# 6) Download model checkpoint (and write checkpoint_info.json)
# -----------------------------
python scripts/download_checkpoint.py \
  --config-id "$MODEL_CONFIG_ID" \
  --out-dir "$ROOT_DIR/_checkpoints"

WEIGHTS_PATH="$(python - <<PY
import json, os
p = os.path.join("$ROOT_DIR","_checkpoints","checkpoint_info.json")
info = json.load(open(p, "r", encoding="utf-8"))
print(info["weights_path"])
PY
)"

echo "[run.sh] WEIGHTS_PATH=$WEIGHTS_PATH"

# -----------------------------
# 7) Build COCO-Spatial-Join-1B dataset
# -----------------------------
python build_coco_spatial_1b.py \
  --coco-root "$COCO_ROOT" \
  --train-json "$COCO_ROOT/annotations/instances_train2017.json" \
  --val-json "$COCO_ROOT/annotations/instances_val2017.json" \
  --output "$OUTPUT_DIR" \
  --model-config-id "$MODEL_CONFIG_ID" \
  --weights "$WEIGHTS_PATH" \
  --seed-root "$SEED_ROOT" \
  --device "$DEVICE" \
  --batch-images "$BATCH_IMAGES" \
  2>&1 | tee "$ROOT_DIR/_logs/build_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[run.sh] DONE. Output at: $OUTPUT_DIR"
