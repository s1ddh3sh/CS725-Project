"""Evaluate a COCO-trained run.

  1. COCO test split (val2017 ids, never trained on): per-source AUC / AP / bAcc / TPR / TNR
  2. Robustness: JPEG / resize / blur applied to every test image
  3. Shortcut tests on REAL test images only: if resizing or re-compressing a real photo makes
     the model call it fake, the model learned processing history instead of generation traces
  4. Synthbuster (optional): cross-dataset test. Real = RAISE, fakes = the 9 generators.
     Reported per generator, from the single real/fake head.

  python evaluate_coco.py --run_dir runs/coco_v1 --synthbuster_root /data/synthbuster
  python evaluate_coco.py ... --sb_long_side 1024    # also test with SYMMETRIC resize of all Synthbuster images
  python evaluate_coco.py ... --sb_jpeg 90           # also test with SYMMETRIC JPEG of all Synthbuster images
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from classes import canonical_class as synthbuster_class
from coco_dataset import EvalDataset, resize_long_side
from coco_engine import detection_report, predict, print_report
from model import AttributionNet
from utils import PERTURBATIONS, load_trainable, make_perturb

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load_run(run_dir, device):
    cfg = json.load(open(Path(run_dir) / "config.json"))
    model = AttributionNet(branches=cfg["branches"], clip_name=cfg["clip_name"], crop=cfg["crop"],
                           lora_r=cfg["lora_r"], lora_last=cfg["lora_last"], branch_drop=0.0,
                           binary_only=True).to(device)
    ck = torch.load(Path(run_dir) / "best.pt", map_location=device)
    load_trainable(model, ck["model"])
    print(f"loaded best.pt (stage {ck['stage']}, epoch {ck['epoch']})")
    return model.eval(), cfg


def recompress_of(cfg):
    rc = cfg.get("recompress")
    return tuple(rc) if rc and rc[1] > 0 else None


def run(model, df, cfg, a, device, perturb=None, real_q=None, long_side=0, grid=None):
    # every evaluation image gets the same deterministic re-encoding the model was trained with
    ds = EvalDataset(df, cfg["crop"], grid or a.grid, perturb=perturb, real_q=real_q, long_side=long_side,
                     recompress=recompress_of(cfg))
    return predict(model, DataLoader(ds, a.batch_size, shuffle=False, num_workers=a.workers), device)


def scan_synthbuster(root):
    rows = []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        c = synthbuster_class(d.name)
        if c is None:
            print(f"[synthbuster] skipping unrecognised folder '{d.name}'")
            continue
        rows += [dict(path=str(p), cls=c, label=int(c != "real")) for p in sorted(d.rglob("*"))
                 if p.suffix.lower() in IMG_EXT]
    df = pd.DataFrame(rows)
    if df.empty or not (df.cls == "real").any():
        raise SystemExit(f"need a real/ (RAISE) folder and generator folders under {root}")
    print(df.cls.value_counts().to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--synthbuster_root", default=None)
    ap.add_argument("--sb_long_side", type=int, default=0,
                    help="extra Synthbuster pass with every image (real and fake) resized to this long side")
    ap.add_argument("--sb_jpeg", type=int, default=0,
                    help="extra Synthbuster pass with every image (real and fake) JPEG-compressed at this quality")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_robustness", action="store_true")
    a = ap.parse_args()

    rd = Path(a.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_run(rd, device)
    real_q = None if cfg.get("no_format_match") else cfg["real_jpeg_q"]
    results = {}

    # COCO test
    te = pd.read_csv(rd / "split_test.csv")
    res = run(model, te, cfg, a, device, real_q=real_q)
    results["coco_test"] = detection_report(res, te.cls.values)
    print_report("COCO test (val2017 ids)", results["coco_test"])

    if not a.skip_robustness:
        results["robustness"] = {}
        for kind, param in PERTURBATIONS:
            if param is None:
                continue
            r = run(model, te, cfg, a, device, perturb=make_perturb(kind, param), real_q=real_q)
            results["robustness"][f"{kind}_{param}"] = detection_report(r, te.cls.values)["overall"]
        print("\nRobustness (all test images perturbed):")
        for k, m in results["robustness"].items():
            print(f"  {k:12s} AUC {m['auc']:.4f}  bAcc {m['bacc']:.4f}  TNR {m['tnr']:.3f}  TPR {m['tpr']:.3f}")

    # shortcut tests
    reals = te[te.cls == "real"].reset_index(drop=True)
    tests = {"clean": None,
             "resize_x0.5": make_perturb("resize", 0.5),
             "long_side_256": lambda im: resize_long_side(im, 256, cfg["crop"]),
             "jpeg_75": make_perturb("jpeg", 75),
             "blur_0.5": make_perturb("blur", 0.5)}
    results["shortcut_fpr_real"] = {}
    print("\nShortcut test - fraction of REAL test photos called fake (should stay near the clean value):")
    for name, f in tests.items():
        r = run(model, reals, cfg, a, device, perturb=f, grid=2)
        fpr = 1 - detection_report(r, reals.cls.values)["overall"]["tnr"]
        results["shortcut_fpr_real"][name] = fpr
        print(f"  {name:22s} FPR {fpr:.3f}")

    # Synthbuster
    if a.synthbuster_root:
        sb = scan_synthbuster(a.synthbuster_root)
        r = run(model, sb, cfg, a, device)   # both sides uncompressed; same re-encoding as training if enabled
        results["synthbuster"] = detection_report(r, sb.cls.values)
        print_report("Synthbuster (cross-dataset, native resolution)", results["synthbuster"])
        if a.sb_long_side:
            r = run(model, sb, cfg, a, device, long_side=a.sb_long_side)
            results[f"synthbuster_long{a.sb_long_side}"] = detection_report(r, sb.cls.values)
            print_report(f"Synthbuster, all images resized to long side {a.sb_long_side}",
                         results[f"synthbuster_long{a.sb_long_side}"])
        if a.sb_jpeg:
        
            r = run(model, sb, cfg, a, device, perturb=make_perturb("jpeg", a.sb_jpeg))
            results[f"synthbuster_jpeg{a.sb_jpeg}"] = detection_report(r, sb.cls.values)
            print_report(f"Synthbuster, every image JPEG q={a.sb_jpeg} (symmetric)",
                         results[f"synthbuster_jpeg{a.sb_jpeg}"])
        print("\nNote: sd13/sd14 share SD 1.5's autoencoder and sdxl shares SDXL-Turbo's, so they are only "
              "partly 'unseen'.\nDALL-E 2/3, Firefly, Midjourney and GLIDE are the real generalisation test.")

    json.dump(results, open(rd / "eval_results.json", "w"), indent=2)
    print(f"\nsaved -> {rd / 'eval_results.json'}")


if __name__ == "__main__":
    main()