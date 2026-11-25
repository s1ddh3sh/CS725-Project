"""End-to-end check of the COCO pipeline on tiny random images. No downloads (branches b,c only).
  python smoke_test_coco.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

tmp = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)


def img(h, w, gray=False):
    a = rng.normal(128, 30, (h, w) if gray else (h, w, 3)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(a)


coco = tmp / "coco"
for split, n, start in (("train_2017", 40, 1), ("val_2017", 10, 1000)):
    (coco / split).mkdir(parents=True)
    for k in range(n):
        cid = start + k
        img(300, 400, gray=(k == 3)).save(coco / split / f"{cid:012d}.jpg", quality=96)
        # DRCT-like fakes: JPEG q75 (as in DRCT-2M), different sizes, nested train/val folders
        for name, size in (("ldm", 256), ("sd15", 320), ("sdxlt", 288)):
            d = tmp / "drct" / name / split
            d.mkdir(parents=True, exist_ok=True)
            img(size, size).save(d / f"{cid:012d}.jpg", quality=75)

sb = tmp / "synthbuster"
for c in ("real", "dalle2", "stable-diffusion-xl", "glide"):
    (sb / c).mkdir(parents=True)
    for k in range(6):
        img(512, 512).save(sb / c / f"{k:04d}.png")

py = sys.executable
subprocess.check_call([py, "build_coco_manifest.py", "--coco_root", str(coco),
                       "--fake", f"ldm={tmp/'drct/ldm'}", "--fake", f"sd15={tmp/'drct/sd15'}",
                       "--fake", f"sdxl_turbo={tmp/'drct/sdxlt'}",
                       "--out", str(tmp / "manifest.csv"), "--n_train_ids", "30", "--n_val_ids", "6"])
subprocess.check_call([py, "train_coco.py", "--manifest", str(tmp / "manifest.csv"), "--out_dir", str(tmp / "run"),
                       "--branches", "bc", "--epochs1", "1", "--epochs2", "0", "--batch_size", "8",
                       "--samples_per_epoch", "32", "--workers", "0", "--val_grid", "1"])
subprocess.check_call([py, "evaluate_coco.py", "--run_dir", str(tmp / "run"), "--synthbuster_root", str(sb),
                       "--sb_long_side", "300", "--workers", "0", "--grid", "1"])
print(f"\nsmoke test passed ({tmp})")