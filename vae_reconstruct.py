#!/usr/bin/env python3
"""
VAE round-trip reconstruction of real images  (aligned real/fake pairs)
=====================================================================
real image -> VAE encoder -> latent -> VAE decoder -> reconstructed "fake"

No denoising is done, so each reconstruction has exactly the same size,
layout and content as its real image; the only difference is the trace the
latent-diffusion autoencoder leaves behind (Rajan et al., ICLR 2025,
"Aligned Datasets Improve Detection of Latent Diffusion-Generated Images").

Output: <out>/<coco_stem>.png   (same 12-digit COCO id as the real file)
        <out>/manifest.csv      (id, real path, recon path, size, crop, PSNR)

Examples
--------
# all of val2017 with the SD-2 VAE
python vae_reconstruct.py --src coco/val2017 --out recon/sd2_val2017 --vae sd2

# a random 20k subset of train2017
python vae_reconstruct.py --src coco/train2017 --out recon/sd2_train2017 --vae sd2 --limit 20000

# only the COCO ids you also have from DRCT (e.g. the SD-1.5 folder you downloaded)
python vae_reconstruct.py --src coco/train2017 --out recon/sd2_train2017 --vae sd2 \
       --match-dir DRCT-2M/stable-diffusion-v1-5/train2017

Requirements:  pip install torch diffusers pillow numpy  (tqdm optional)
"""
import argparse
import csv
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# name -> (HF repo, subfolder, safe to run in fp16)
# The SD-1.x VAE is shared by SD 1.4 and SD 1.5. The original SDXL VAE produces
# NaNs in fp16, so it runs in fp32 here (slower, but the exact SDXL fingerprint).
VAE_PRESETS = {
    "sd1": ("CompVis/stable-diffusion-v1-4", "vae", True),
    # Stability AI removed the official SD 2.x repos from Hugging Face; this is a
    # Hub "duplicate" of stabilityai/stable-diffusion-2-1 (weights stored as .bin).
    "sd2": ("philschmid/stable-diffusion-2-1", "vae", True),
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", "vae", False),
}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with real images (e.g. coco/train2017)")
    ap.add_argument("--out", required=True, help="output folder for reconstructions")
    ap.add_argument("--vae", default="sd2",
                    help=f"preset ({', '.join(VAE_PRESETS)}) or 'repo' / 'repo:subfolder' on Hugging Face")
    ap.add_argument("--batch-size", type=int, default=8, help="lower to 2-4 on a 16 GB T4 if you get OOM")
    ap.add_argument("--workers", type=int, default=4, help="image-loading processes")
    ap.add_argument("--save-workers", type=int, default=4, help="image-saving threads")
    ap.add_argument("--limit", type=int, default=0, help="random subset of N images (0 = all)")
    ap.add_argument("--match-dir", default=None,
                    help="only process images whose COCO id also appears in this folder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true",
                    help="sample the latent instead of using its mean (default: mean, deterministic)")
    ap.add_argument("--jpeg-quality", type=int, default=0,
                    help="save reconstructions as JPEG at this quality instead of PNG (0 = PNG). "
                         "Usually better: keep PNG and apply random JPEG to BOTH classes at train time.")
    ap.add_argument("--keep-gray", action="store_true",
                    help="also process grayscale images (skipped by default: the VAE adds colour casts)")
    ap.add_argument("--fp32", action="store_true", help="force fp32 even on GPU")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def coco_id(path: Path) -> str:
    """'000000123456.jpg' -> '000000123456' (keeps the zero padding)."""
    return path.stem


def scan(args, out_dir: Path, ext: str):
    files = sorted(p for p in Path(args.src).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"No images found in {args.src}")

    if args.match_dir:
        wanted = {coco_id(p) for p in Path(args.match_dir).rglob("*") if p.suffix.lower() in IMG_EXTS}
        files = [p for p in files if coco_id(p) in wanted]
        print(f"--match-dir: {len(files)} real images have a counterpart in {args.match_dir}")

    if args.limit and args.limit < len(files):
        random.Random(args.seed).shuffle(files)
        files = sorted(files[: args.limit])

    items, skipped = [], defaultdict(int)
    for p in files:
        if not args.overwrite and (out_dir / f"{coco_id(p)}{ext}").exists():
            skipped["already done"] += 1
            continue
        with Image.open(p) as im:  # reads the header only
            w, h = im.size
            mode = im.mode
        if mode not in ("RGB", "RGBA", "P", "CMYK", "YCbCr") and not args.keep_gray:
            skipped["grayscale"] += 1
            continue
        # The VAE downsamples by 8, so both sides must be multiples of 8.
        # 640x480 needs no crop; other COCO sizes lose at most 7 px on the
        # right/bottom. The crop is recorded so the real image can be cut identically.
        w8, h8 = w - w % 8, h - h % 8
        if min(w8, h8) < 64:
            skipped["too small"] += 1
            continue
        items.append((str(p), w8, h8, w, h))
    return items, dict(skipped)


class RealImages(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, w8, h8, _, _ = self.items[i]
        with Image.open(path) as im:
            im = im.convert("RGB").crop((0, 0, w8, h8))
            arr = np.asarray(im, dtype=np.uint8).copy()
        return torch.from_numpy(arr).permute(2, 0, 1), i  # uint8, C x H x W


def size_batches(items, bs):
    """Images can only be batched with others of the same size."""
    buckets = defaultdict(list)
    for i, it in enumerate(items):
        buckets[(it[1], it[2])].append(i)
    return [idx[k:k + bs] for idx in buckets.values() for k in range(0, len(idx), bs)]


def load_vae(name, device, want_fp16):
    from diffusers import AutoencoderKL

    if name in VAE_PRESETS:
        repo, sub, fp16_ok = VAE_PRESETS[name]
    else:
        repo, _, sub = name.partition(":")
        sub, fp16_ok = (sub or None), True
    dtype = torch.float16 if (want_fp16 and fp16_ok) else torch.float32
    try:
        vae = AutoencoderKL.from_pretrained(repo, subfolder=sub, torch_dtype=dtype)
    except Exception as e:
        raise SystemExit(f"Could not load VAE from '{repo}' (subfolder={sub}): {e}\n"
                         "If the repo moved, pass another copy with --vae 'repo:subfolder'.")
    vae = vae.to(device).eval()
    vae.enable_slicing()  # decode one image at a time -> much lower peak memory
    print(f"VAE: {repo}{'/' + sub if sub else ''}  dtype={dtype}  device={device}")
    return vae, dtype


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: no GPU found - this will take seconds per image.")
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jpg" if args.jpeg_quality else ".png"

    items, skipped = scan(args, out_dir, ext)
    print(f"To process: {len(items)}   skipped: {skipped or 'none'}")
    if not items:
        return

    vae, dtype = load_vae(args.vae, device, want_fp16=(device == "cuda" and not args.fp32))
    loader = DataLoader(RealImages(items), batch_sampler=size_batches(items, args.batch_size),
                        num_workers=args.workers, pin_memory=(device == "cuda"),
                        persistent_workers=args.workers > 0)
    gen = torch.Generator(device=device).manual_seed(args.seed) if args.sample else None

    # bounded background saving so the GPU never waits for PNG encoding
    pool = ThreadPoolExecutor(args.save_workers)
    slots = threading.BoundedSemaphore(args.save_workers * 8)

    def save(arr, path):
        try:
            im = Image.fromarray(arr)
            if args.jpeg_quality:
                im.save(path, quality=args.jpeg_quality)
            else:
                im.save(path, compress_level=1)  # lossless, fast
        finally:
            slots.release()

    manifest = out_dir / "manifest.csv"
    new_file = not manifest.exists()
    mf = open(manifest, "a", newline="")
    writer = csv.writer(mf)
    if new_file:
        writer.writerow(["coco_id", "real_path", "recon_path", "orig_w", "orig_h",
                         "crop_w", "crop_h", "psnr_db"])

    try:
        from tqdm import tqdm
        bar = tqdm(total=len(items), unit="img")
    except ImportError:
        bar = None

    done, t0, psnrs = 0, time.time(), []
    with torch.inference_mode():
        for x_u8, idx in loader:
            x_u8 = x_u8.to(device, non_blocking=True)
            x = x_u8.to(dtype) / 127.5 - 1.0                      # [-1, 1]
            dist = vae.encode(x).latent_dist
            z = dist.sample(generator=gen) if args.sample else dist.mode()
            y = vae.decode(z).sample.float().clamp(-1, 1)
            y_u8 = ((y + 1.0) * 127.5).round().to(torch.uint8)

            mse = (y_u8.float() - x_u8.float()).pow(2).flatten(1).mean(1).clamp_min(1e-10)
            psnr = (10 * torch.log10(255.0 ** 2 / mse)).tolist()
            arrs = y_u8.permute(0, 2, 3, 1).cpu().numpy()

            for k, i in enumerate(idx.tolist()):
                path, w8, h8, w, h = items[i]
                cid = coco_id(Path(path))
                rec_path = out_dir / f"{cid}{ext}"
                slots.acquire()
                pool.submit(save, arrs[k], rec_path)
                writer.writerow([cid, path, str(rec_path), w, h, w8, h8, f"{psnr[k]:.2f}"])
                psnrs.append(psnr[k])

            mf.flush()
            done += len(idx)
            if bar:
                bar.update(len(idx))
            elif done % (args.batch_size * 50) < args.batch_size:
                rate = done / (time.time() - t0)
                print(f"{done}/{len(items)}  {rate:.1f} img/s  ETA {(len(items) - done) / rate / 60:.1f} min")

    pool.shutdown(wait=True)
    mf.close()
    if bar:
        bar.close()
    dt = time.time() - t0
    print(f"Done: {done} images in {dt / 60:.1f} min ({done / dt:.1f} img/s). "
          f"PSNR mean {np.mean(psnrs):.2f} dB, min {np.min(psnrs):.2f} dB  -> {manifest}")
    print("Sanity check: SD VAEs usually give roughly 25-35 dB on photos. Far lower = something is wrong "
          "(colour order, value range); 50+ dB = the image barely changed.")


if __name__ == "__main__":
    main()
