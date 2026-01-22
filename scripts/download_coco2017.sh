#!/usr/bin/env bash
set -euo pipefail

# Download MS COCO 2017 train/val images + detection annotations.
#
# This script supports two sources:
#   1) Hugging Face mirror (default): faster/more stable in many regions
#   2) Official COCO website: http://images.cocodataset.org
#
# Usage:
#   bash scripts/download_coco2017.sh /path/to/_coco2017
#
# Env overrides:
#   COCO_DOWNLOAD_SOURCE=hf|official|auto
#   HF_COCO_REPO=pcuenq/coco-2017-mirror
#   HF_COCO_REVISION=main
#
# Hugging Face acceleration knobs (optional):
#   HF_HUB_ENABLE_HF_TRANSFER=1    # enable hf_transfer (Rust fast path)
#   HF_XET_HIGH_PERFORMANCE=1      # speed up hf_xet when repo uses Xet storage
#   HF_HOME=/path/to/hf_cache      # where HF stores cache

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/coco2017_root"
  exit 1
fi

COCO_ROOT="$1"
mkdir -p "$COCO_ROOT"
mkdir -p "$COCO_ROOT/_downloads"

SOURCE="${COCO_DOWNLOAD_SOURCE:-hf}"  # hf|official|auto

# Hugging Face mirror settings
HF_COCO_REPO="${HF_COCO_REPO:-pcuenq/coco-2017-mirror}"
HF_COCO_REVISION="${HF_COCO_REVISION:-main}"

TRAIN_ZIP_NAME="train2017.zip"
VAL_ZIP_NAME="val2017.zip"
ANN_ZIP_NAME="annotations_trainval2017.zip"

TRAIN_ZIP_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_ZIP_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_ZIP_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

download_official() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "[download_coco2017.sh] Exists: $out"
    return
  fi
  echo "[download_coco2017.sh] (official) Downloading: $url"
  # -C - enables resume if the server supports it.
  curl -L --fail -C - -o "$out" "$url"
}

zip_is_valid() {
  local zip_path="$1"
  # `unzip -tq` is a quick integrity test (fails fast on truncated downloads).
  unzip -tq "$zip_path" >/dev/null 2>&1
}

ensure_valid_or_remove() {
  local zip_path="$1"
  if [[ -f "$zip_path" ]]; then
    if zip_is_valid "$zip_path"; then
      echo "[download_coco2017.sh] Valid zip: $zip_path"
    else
      echo "[download_coco2017.sh] Corrupted/incomplete zip detected, removing: $zip_path"
      rm -f "$zip_path"
    fi
  fi
}

download_hf() {
  echo "[download_coco2017.sh] (hf) Using mirror repo: ${HF_COCO_REPO} (revision=${HF_COCO_REVISION})"

  if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" == "1" ]]; then
    echo "[download_coco2017.sh] (hf) HF_HUB_ENABLE_HF_TRANSFER=1 (hf_transfer fast path enabled if available)"
  fi
  if [[ "${HF_XET_HIGH_PERFORMANCE:-0}" == "1" ]]; then
    echo "[download_coco2017.sh] (hf) HF_XET_HIGH_PERFORMANCE=1 (xet high-performance mode)"
  fi
  if [[ -n "${HF_HOME:-}" ]]; then
    echo "[download_coco2017.sh] (hf) HF_HOME=${HF_HOME}"
  fi

  # Prefer the modern `hf download` CLI shipped with huggingface_hub.
  if command -v hf >/dev/null 2>&1; then
    # Some versions may not recognize all flags; keep invocation minimal.
    if hf download "$HF_COCO_REPO" "$TRAIN_ZIP_NAME" "$VAL_ZIP_NAME" "$ANN_ZIP_NAME" \
      --repo-type dataset \
      --revision "$HF_COCO_REVISION" \
      --local-dir "$COCO_ROOT/_downloads"; then
      return
    fi
    echo "[download_coco2017.sh] (hf) 'hf download' failed; trying 'huggingface-cli download' as fallback."
  fi

  # Fallback to legacy `huggingface-cli` (still widely available).
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$HF_COCO_REPO" \
      --repo-type dataset \
      --revision "$HF_COCO_REVISION" \
      --include "$TRAIN_ZIP_NAME" --include "$VAL_ZIP_NAME" --include "$ANN_ZIP_NAME" \
      --local-dir "$COCO_ROOT/_downloads" || return 1
    return
  fi

  # Last resort: use Python API.
  python - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = os.environ.get("HF_COCO_REPO", "pcuenq/coco-2017-mirror")
revision = os.environ.get("HF_COCO_REVISION", "main")
out_dir = os.path.join(os.environ["COCO_ROOT"], "_downloads")
os.makedirs(out_dir, exist_ok=True)

for fname in ["train2017.zip", "val2017.zip", "annotations_trainval2017.zip"]:
    print(f"[download_coco2017.sh] (hf/python) Downloading {fname} from {repo_id}@{revision}")
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=fname,
        revision=revision,
        local_dir=out_dir,
        local_dir_use_symlinks=True,  # avoid duplicating huge files
    )
    print(f"[download_coco2017.sh] (hf/python) -> {path}")
PY
}

unzip_if_needed() {
  local zip_path="$1"
  local out_dir="$2"
  local marker="$3"
  if [[ -e "$marker" ]]; then
    echo "[download_coco2017.sh] Already extracted: $marker"
    return
  fi
  echo "[download_coco2017.sh] Extracting: $zip_path -> $out_dir"
  unzip -q "$zip_path" -d "$out_dir"
}

ensure_downloads_present() {
  test -f "$COCO_ROOT/_downloads/$TRAIN_ZIP_NAME"
  test -f "$COCO_ROOT/_downloads/$VAL_ZIP_NAME"
  test -f "$COCO_ROOT/_downloads/$ANN_ZIP_NAME"
}

try_hf_then_official() {
  if download_hf; then
    return
  fi
  echo "[download_coco2017.sh] (hf) Failed; falling back to official COCO URLs."
  download_official "$TRAIN_ZIP_URL" "$COCO_ROOT/_downloads/$TRAIN_ZIP_NAME"
  download_official "$VAL_ZIP_URL"   "$COCO_ROOT/_downloads/$VAL_ZIP_NAME"
  download_official "$ANN_ZIP_URL"   "$COCO_ROOT/_downloads/$ANN_ZIP_NAME"
}

export COCO_ROOT
export HF_COCO_REPO
export HF_COCO_REVISION

# If a previous download was interrupted (common for 18GB+ files), the zip might
# exist but be invalid. We proactively detect & delete such partial files so that
# re-download works smoothly (HF or official).
ensure_valid_or_remove "$COCO_ROOT/_downloads/$TRAIN_ZIP_NAME"
ensure_valid_or_remove "$COCO_ROOT/_downloads/$VAL_ZIP_NAME"
ensure_valid_or_remove "$COCO_ROOT/_downloads/$ANN_ZIP_NAME"

case "$SOURCE" in
  hf)
    download_hf || {
      echo "[download_coco2017.sh] (hf) Download failed. You can retry, or set COCO_DOWNLOAD_SOURCE=official."
      exit 1
    }
    ;;
  official)
    download_official "$TRAIN_ZIP_URL" "$COCO_ROOT/_downloads/$TRAIN_ZIP_NAME"
    download_official "$VAL_ZIP_URL"   "$COCO_ROOT/_downloads/$VAL_ZIP_NAME"
    download_official "$ANN_ZIP_URL"   "$COCO_ROOT/_downloads/$ANN_ZIP_NAME"
    ;;
  auto)
    try_hf_then_official
    ;;
  *)
    echo "[download_coco2017.sh] Unknown COCO_DOWNLOAD_SOURCE='$SOURCE' (expected hf|official|auto)"
    exit 1
    ;;
esac

ensure_downloads_present

unzip_if_needed "$COCO_ROOT/_downloads/$TRAIN_ZIP_NAME" "$COCO_ROOT" "$COCO_ROOT/train2017"
unzip_if_needed "$COCO_ROOT/_downloads/$VAL_ZIP_NAME"   "$COCO_ROOT" "$COCO_ROOT/val2017"
unzip_if_needed "$COCO_ROOT/_downloads/$ANN_ZIP_NAME"   "$COCO_ROOT" "$COCO_ROOT/annotations/instances_train2017.json"

# Quick existence checks:
test -d "$COCO_ROOT/train2017"
test -d "$COCO_ROOT/val2017"
test -f "$COCO_ROOT/annotations/instances_train2017.json"
test -f "$COCO_ROOT/annotations/instances_val2017.json"

echo "[download_coco2017.sh] OK: COCO_ROOT=$COCO_ROOT (source=$SOURCE)"
