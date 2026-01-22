#!/usr/bin/env bash
set -euo pipefail

# Install detectron2.
#
# Recommended (default): build from source from GitHub:
#   DETECTRON2_GIT_REF=main (or a commit hash)
#
# Optional: install a pre-built wheel index if you have a matching wheel URL:
#   DETECTRON2_WHEEL_INDEX=https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.4/index.html

DETECTRON2_GIT_REF="${DETECTRON2_GIT_REF:-main}"
DETECTRON2_WHEEL_INDEX="${DETECTRON2_WHEEL_INDEX:-}"

echo "[install_detectron2.sh] DETECTRON2_GIT_REF=$DETECTRON2_GIT_REF"

if [[ -n "$DETECTRON2_WHEEL_INDEX" ]]; then
  echo "[install_detectron2.sh] Installing detectron2 from wheel index (user-specified):"
  echo "  $DETECTRON2_WHEEL_INDEX"
  python -m pip install "detectron2" -f "$DETECTRON2_WHEEL_INDEX"
else
  # Best-effort: auto-select official wheel index based on torch+cuda version.
  # If wheel install fails (no matching wheel / network), fall back to source install.
  AUTO_INDEX="$(python - <<'PY'
import re
try:
    import torch
    tv = torch.__version__
    m = re.match(r"^(\d+)\.(\d+)", tv)
    if not m:
        raise RuntimeError("cannot parse torch version")
    torch_mm = f"{m.group(1)}.{m.group(2)}"
    cu = getattr(torch.version, "cuda", None)
    if not cu:
        print("")
    else:
        # torch.version.cuda is like '11.8' or '12.1'
        if cu.startswith("11."):
            tag = "cu118"
        elif cu.startswith("12."):
            tag = "cu121"
        else:
            tag = "cu121"
        print(f"https://dl.fbaipublicfiles.com/detectron2/wheels/{tag}/torch{torch_mm}/index.html")
except Exception:
    print("")
PY
)"

  if [[ -n "$AUTO_INDEX" ]]; then
    echo "[install_detectron2.sh] Trying detectron2 wheel index (auto):"
    echo "  $AUTO_INDEX"
    if python -m pip install "detectron2" -f "$AUTO_INDEX"; then
      echo "[install_detectron2.sh] detectron2 wheel install OK"
    else
      echo "[install_detectron2.sh] Wheel install failed; falling back to GitHub source (will compile extensions)."
      echo "[install_detectron2.sh] If this fails, ensure you have build tools (gcc/g++, cmake) and (for CUDA) nvcc."
      # NOTE:
      # Detectron2's build step imports torch. With pip's default PEP517 build isolation,
      # the temporary build env often does NOT include torch, causing:
      #   ModuleNotFoundError: No module named 'torch'
      # Therefore we disable build isolation so the build can import the already-installed torch.
      python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_GIT_REF}"
    fi
  else
    echo "[install_detectron2.sh] Could not auto-detect a wheel index; installing from GitHub source (will compile extensions)."
    echo "[install_detectron2.sh] If this fails, ensure you have build tools (gcc/g++, cmake) and (for CUDA) nvcc."
    python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_GIT_REF}"
  fi
fi

python - <<'PY'
import detectron2
print("[install_detectron2.sh] detectron2:", getattr(detectron2, "__version__", "unknown"))
PY
