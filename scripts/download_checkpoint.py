#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True, help="Detectron2 model zoo config id, e.g. COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--out-dir", required=True, help="Directory to store downloaded checkpoint + metadata")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detectron2 model zoo utilities
    from detectron2 import model_zoo
    from detectron2.utils.file_io import PathManager

    config_id = args.config_id
    ckpt_url = model_zoo.get_checkpoint_url(config_id)

    # This downloads into detectron2 cache and returns a local file path
    local_cached = PathManager.get_local_path(ckpt_url)
    local_cached = Path(local_cached)

    weights_path = out_dir / local_cached.name
    if not weights_path.exists():
        shutil.copyfile(local_cached, weights_path)

    ckpt_sha256 = sha256_file(str(weights_path))

    info = {
        "model_config_id": config_id,
        "checkpoint_url": ckpt_url,
        "weights_path": str(weights_path.resolve()),
        "checkpoint_sha256": ckpt_sha256,
    }

    (out_dir / "checkpoint_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
