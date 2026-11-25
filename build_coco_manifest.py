"""Build a paired, leakage-free manifest: COCO real images + DRCT-2M fakes (+ optional reconstructions).

Every selected COCO id contributes its real image AND one image from every fake source,
so the content distribution is identical across classes. Splits are made by COCO id:
    train / val : ids from COCO train2017
    test        : ids from COCO val2017 (never seen in training)

  python build_coco_manifest.py --coco_root /data/coco \
      --fake ldm=/data/DRCT-2M/ldm-text2im-large-256 \
      --fake sd15=/data/DRCT-2M/stable-diffusion-v1-5 \
      --fake sdxl_turbo=/data/DRCT-2M/sdxl-turbo \
      --out /data/coco_manifest.csv --n_train_ids 20000 --n_val_ids 1000

  # later, add VAE reconstructions as a separate family:
      --recon vae_sd2=/data/recon/sd2_train  --recon vae_sd2_test=...   (any folder, matched by id)

Writes <out>.csv (one row per image) and <out>.meta.json (class list, families,
JPEG qualities of the real images, used to format-match PNG fakes at train time).
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
FAMILIES = ["real", "text2img", "recon"]

# IJG standard luminance quantisation table (quality 50)
_STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
             14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
             18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
             49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def jpeg_quality(im):
    """Approximate IJG quality from the luminance table (order-independent). None if not JPEG."""
    q = getattr(im, "quantization", None)
    if not q or 0 not in q:
        return None
    scale = 100.0 * sum(q[0]) / sum(_STD_LUMA)
    quality = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return int(round(min(100, max(1, quality))))


def coco_id(path: Path):
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else None


def index_dir(d: Path):
    idx, dups = {}, 0
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            i = coco_id(p)
            if i is None:
                continue
            if i in idx:
                dups += 1
                continue
            idx[i] = p
    if not idx:
        raise SystemExit(f"No images found in {d}")
    if dups:
        print(f"  WARNING: {dups} duplicate ids in {d} (kept the first of each)")
    return idx


def split_dir(root: Path, split: str):
    """COCO split folder: accepts train2017 / train_2017 (and val2017 / val_2017)."""
    for name in (f"{split}2017", f"{split}_2017"):
        if (root / name).is_dir():
            return root / name
    raise SystemExit(f"Could not find {split}2017/ or {split}_2017/ inside {root}")


def header(p):
    with Image.open(p) as im:
        return dict(mode=im.mode, w=im.size[0], h=im.size[1], fmt=im.format, jpeg_q=jpeg_quality(im))


def parse_sources(items, kind):
    out = []
    for s in items or []:
        name, _, path = s.partition("=")
        if not path:
            raise SystemExit(f"--{kind} expects NAME=DIR, got '{s}'")
        out.append((name.strip(), Path(path.strip())))
    return out


def select(ids, n, real_idx, src_idx, rng, args, stats):
    """Shuffle ids and keep the first n whose real image is usable (colour, big enough)."""
    ids = sorted(ids)
    rng.shuffle(ids)
    keep, heads = [], {}
    for i in ids:
        if n and len(keep) >= n:
            break
        h = header(real_idx[i])
        if h["mode"] != "RGB" and not args.keep_gray:   # grayscale COCO photos: colour is a shortcut
            stats["real not RGB"] += 1
            continue
        if min(h["w"], h["h"]) < args.min_side:          # would need padding -> padding is a shortcut
            stats["real too small"] += 1
            continue
        fh = {name: header(idx[i]) for name, idx in src_idx.items()}
        if any(min(v["w"], v["h"]) < args.min_side for v in fh.values()):
            stats["fake too small"] += 1
            continue
        keep.append(i)
        heads[i] = (h, fh)
    return keep, heads


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco_root", required=True, help="folder containing train2017/ (or train_2017/) and val2017/ (or val_2017/)")
    ap.add_argument("--fake", action="append", help="NAME=DIR of a text-to-image fake set (repeatable)")
    ap.add_argument("--recon", action="append", help="NAME=DIR of a reconstruction set (repeatable)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--n_train_ids", type=int, default=20000)
    ap.add_argument("--n_val_ids", type=int, default=1000)
    ap.add_argument("--n_test_ids", type=int, default=0, help="0 = every usable val2017 id")
    ap.add_argument("--min_side", type=int, default=224, help="drop ids where any image is smaller than the crop")
    ap.add_argument("--keep_gray", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    fakes, recons = parse_sources(a.fake, "fake"), parse_sources(a.recon, "recon")
    if not fakes and not recons:
        raise SystemExit("give at least one --fake or --recon")
    names = [n for n, _ in fakes + recons]
    if len(set(names)) != len(names) or "real" in names:
        raise SystemExit(f"source names must be unique and not 'real': {names}")
    classes = ["real"] + names
    families = {"real": 0, **{n: 1 for n, _ in fakes}, **{n: 2 for n, _ in recons}}

    coco = Path(a.coco_root)
    print("indexing COCO ...")
    real_tr, real_te = index_dir(split_dir(coco, "train")), index_dir(split_dir(coco, "val"))
    src_idx = {}
    for name, d in fakes + recons:
        print(f"indexing {name} <- {d}")
        src_idx[name] = index_dir(d)
        n_tr = len(src_idx[name].keys() & real_tr.keys())
        n_te = len(src_idx[name].keys() & real_te.keys())
        print(f"  {len(src_idx[name])} files; matches {n_tr} train2017 ids and {n_te} val2017 ids")

    common = set.intersection(*(set(v) for v in src_idx.values()))
    rng = random.Random(a.seed)
    stats = Counter()
    tr_ids, tr_heads = select(common & real_tr.keys(), a.n_train_ids + a.n_val_ids, real_tr, src_idx, rng, a, stats)
    te_ids, te_heads = select(common & real_te.keys(), a.n_test_ids, real_te, src_idx, rng, a, stats)
    if len(tr_ids) < a.n_train_ids + a.n_val_ids:
        print(f"WARNING: only {len(tr_ids)} usable train2017 ids (asked {a.n_train_ids + a.n_val_ids})")
    n_val = min(a.n_val_ids, len(tr_ids) // 10 or 1)
    split_of = {i: "val" for i in tr_ids[:n_val]}
    split_of.update({i: "train" for i in tr_ids[n_val:]})
    split_of.update({i: "test" for i in te_ids})

    rows = []
    for ids, real_idx, heads in ((tr_ids, real_tr, tr_heads), (te_ids, real_te, te_heads)):
        for i in ids:
            rh, fh = heads[i]
            rows.append(dict(path=str(real_idx[i]), cls="real", family=0, coco_id=i, split=split_of[i], **rh))
            for name in names:
                rows.append(dict(path=str(src_idx[name][i]), cls=name, family=families[name], coco_id=i,
                                 split=split_of[i], **fh[name]))
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    real_q = df[(df.cls == "real") & (df.split == "train")].jpeg_q.dropna().astype(int).tolist()
    meta = dict(classes=classes, families=families, family_names=FAMILIES, real_jpeg_q=real_q,
                sources={n: str(d) for n, d in fakes + recons}, coco_root=str(coco), seed=a.seed)
    json.dump(meta, open(out.with_suffix(".meta.json"), "w"), indent=1)

    # ------------------------------------------------------------------ audit
    print(f"\nskipped ids: {dict(stats) or 'none'}")
    print("\nimages per split / class:")
    print(df.pivot_table(index="cls", columns="split", values="path", aggfunc="count").fillna(0).astype(int))
    print("\nformat / size audit (look for differences between real and fakes):")
    for c in classes:
        d = df[df.cls == c]
        sizes = Counter(zip(d.w, d.h)).most_common(3)
        q = d.jpeg_q.dropna()
        qtxt = f"  JPEG q median {q.median():.0f}" if len(q) else ""
        print(f"  {c:12s} formats {dict(Counter(d.fmt))}  modes {dict(Counter(d['mode']))}  top sizes {sizes}{qtxt}")
    jpeg_fakes = [c for c in classes if c != "real" and (df[df.cls == c].fmt == "JPEG").any()]
    if jpeg_fakes:
        print(f"\nNOTE: {jpeg_fakes} are already JPEG. If their JPEG quality (above) differs from the real "
              f"images', quality alone can separate the classes. train_coco.py handles this by default: "
              f"--recompress 60 70 re-encodes EVERY image (both classes) below both original qualities.")
    if real_q:
        print(f"\nreal JPEG quality: median {np.median(real_q):.0f}, 5-95% "
              f"[{np.percentile(real_q, 5):.0f}, {np.percentile(real_q, 95):.0f}]")
    png_fakes = df[(df.cls != "real") & (df.fmt != "JPEG")].cls.unique().tolist()
    if png_fakes:
        print(f"non-JPEG fakes {png_fakes}: training will JPEG them with qualities drawn from the real images "
              f"(format matching), so 'JPEG = real' cannot be learned.")
    print(f"\nwrote {out} ({len(df)} rows) and {out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()