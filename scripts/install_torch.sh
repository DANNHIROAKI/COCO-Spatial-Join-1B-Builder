#!/usr/bin/env bash
set -euo pipefail

# Install a known-good PyTorch + torchvision pair.
# You can override versions & CUDA flavor via:
#   TORCH_CUDA=cu121|cu118|cpu
#   TORCH_VERSION=2.4.1
#   TORCHVISION_VERSION=0.19.1
#
# Notes:
# - For CUDA wheels, PyTorch uses a separate index.
# - This script is intended for Linux. If you're on macOS/Windows, you should
#   set TORCH_CUDA=cpu and potentially remove the +cpu suffix manually.

TORCH_CUDA="${TORCH_CUDA:-auto}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"

if [[ "$TORCH_CUDA" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    # Prefer matching the *local CUDA toolkit* when available, because
    # detectron2 may compile CUDA extensions (nvcc) and mismatches can break.
    # - If nvcc is present and reports CUDA 11.x -> use cu118 wheels.
    # - If nvcc reports CUDA 12.x -> use cu121 wheels.
    # - If nvcc missing -> default to cu121 (user can override).
    if command -v nvcc >/dev/null 2>&1; then
      NVCC_REL=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)
      case "$NVCC_REL" in
        11.*)
          TORCH_CUDA="cu118"
          ;;
        12.*)
          TORCH_CUDA="cu121"
          ;;
        *)
          TORCH_CUDA="cu121"
          ;;
      esac
      echo "[install_torch.sh] Detected nvcc release: ${NVCC_REL:-unknown} -> TORCH_CUDA=$TORCH_CUDA"
    else
      TORCH_CUDA="cu121"
      echo "[install_torch.sh] nvcc not found; defaulting TORCH_CUDA=$TORCH_CUDA (override via TORCH_CUDA=cu118|cu121|cpu)"
    fi
  else
    TORCH_CUDA="cpu"
  fi
fi

echo "[install_torch.sh] TORCH_CUDA=$TORCH_CUDA"
echo "[install_torch.sh] TORCH_VERSION=$TORCH_VERSION"
echo "[install_torch.sh] TORCHVISION_VERSION=$TORCHVISION_VERSION"

if [[ "$TORCH_CUDA" == "cpu" ]]; then
  python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VERSION}+cpu" \
    "torchvision==${TORCHVISION_VERSION}+cpu"
elif [[ "$TORCH_CUDA" == "cu118" ]]; then
  python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    "torch==${TORCH_VERSION}+cu118" \
    "torchvision==${TORCHVISION_VERSION}+cu118"
elif [[ "$TORCH_CUDA" == "cu121" ]]; then
  python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==${TORCH_VERSION}+cu121" \
    "torchvision==${TORCHVISION_VERSION}+cu121"
else
  echo "[install_torch.sh] ERROR: unsupported TORCH_CUDA=$TORCH_CUDA (expected cu121|cu118|cpu|auto)"
  exit 1
fi

python - <<'PY'
import torch, torchvision
print("[install_torch.sh] torch:", torch.__version__)
print("[install_torch.sh] torchvision:", torchvision.__version__)
print("[install_torch.sh] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[install_torch.sh] cuda version:", torch.version.cuda)
    print("[install_torch.sh] device:", torch.cuda.get_device_name(0))
PY
