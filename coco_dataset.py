"""Datasets for the COCO + DRCT-2M setup.

Shortcut controls (all applied identically to every class, except format matching,
which exists precisely to remove a class difference):
  * re-compression  : EVERY image (real and fake) is re-encoded as JPEG at a quality drawn from
                      `recompress` (e.g. 60-70) before anything else. DRCT fakes are stored as JPEG
                      q=75 and COCO reals at q~96; without this, the JPEG quality alone (different
                      quantisation tables) separates the classes. Re-encoding both at a LOWER quality
                      than either original gives them the same final tables.
  * format matching : (only if recompress is off) non-JPEG fakes are JPEG-compressed once, with a
                      quality drawn from the real COCO images
  * no fixed resize : random crops at native resolution
  * random rescale  : same distribution for both classes, so resampling is not a label cue and the
                      field-of-view difference (256-px LDM vs 640-px COCO) is blurred out
  * blur / JPEG aug : same for both classes
  * balanced sampler: 50% real, 50% fake split evenly over the fake sources
"""
import io
import random
import zlib

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, WeightedRandomSampler

Image.MAX_IMAGE_PIXELS = None          # RAISE images are large
FILTERS = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]


def jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rand_rescale(img, crop, lo, hi):
    w, h = img.size
    s = max(random.uniform(lo, hi), crop / min(w, h))      # never smaller than the crop
    if abs(s - 1.0) < 1e-3:
        return img
    return img.resize((max(crop, round(w * s)), max(crop, round(h * s))), random.choice(FILTERS))


def resize_long_side(img, long_side, crop):
    """Symmetric eval-time resize: longest side -> long_side (short side kept >= crop)."""
    w, h = img.size
    s = max(long_side / max(w, h), crop / min(w, h))
    return img.resize((max(crop, round(w * s)), max(crop, round(h * s))), Image.BICUBIC)


def _pad_to(arr, size):
    h, w = arr.shape[:2]
    ph, pw = max(0, size - h), max(0, size - w)
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return arr


def _to_tensor(arr):
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().div_(255.0)


def _grid(n, c, g):
    if g == 1:
        return [(n - c) // 2]
    return np.linspace(0, n - c, g).round().astype(int).tolist()


def needs_format_match(df):
    """Fakes that are not JPEG while the reals are."""
    if "fmt" not in df:
        return [False] * len(df)
    return ((df.cls != "real") & (df.fmt.fillna("") != "JPEG")).tolist()


class TrainDataset(Dataset):
    """Returns (crop, y) with y = 1 for fake, 0 for real."""

    def __init__(self, df, crop=224, aug=True, real_q=None, match_jpeg=True, recompress=None,
                 p_rescale=0.5, scale=(0.6, 1.4), p_blur=0.2, p_jpeg=0.3, jpeg_range=(65, 100)):
        self.recompress = tuple(recompress) if recompress else None
        if self.recompress:
            match_jpeg = False                      # every image is re-encoded anyway
        self.paths = df.path.tolist()
        self.labels = (df.cls != "real").astype(int).tolist()
        self.match = needs_format_match(df) if (match_jpeg and real_q) else [False] * len(df)
        self.real_q = list(real_q or [])
        self.crop, self.aug = crop, aug
        self.p_rescale, self.scale = p_rescale, scale
        self.p_blur, self.p_jpeg, self.jpeg_range = p_blur, p_jpeg, jpeg_range

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.recompress:
            img = jpeg(img, random.randint(*self.recompress))
        elif self.match[i]:
            img = jpeg(img, random.choice(self.real_q))
        if self.aug and random.random() < self.p_rescale:
            img = rand_rescale(img, self.crop, *self.scale)
        arr = _pad_to(np.asarray(img), self.crop)
        h, w = arr.shape[:2]
        y0, x0 = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
        patch = Image.fromarray(arr[y0:y0 + self.crop, x0:x0 + self.crop])
        if self.aug:
            if random.random() < 0.5:
                patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < self.p_blur:
                patch = patch.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 1.2)))
            if random.random() < self.p_jpeg:
                patch = jpeg(patch, random.randint(*self.jpeg_range))
        return _to_tensor(np.asarray(patch)), self.labels[i]


class EvalDataset(Dataset):
    """Deterministic g x g grid of crops -> [g*g, 3, crop, crop].
    `perturb` is any PIL->PIL callable. Format matching is deterministic (quality chosen by path hash)."""

    def __init__(self, df, crop=224, grid=2, perturb=None, real_q=None, match_jpeg=True, long_side=0,
                 label_col="label", recompress=None):
        self.recompress = tuple(recompress) if recompress else None
        if self.recompress:
            match_jpeg = False
        self.paths, self.labels = df.path.tolist(), df[label_col].tolist()
        self.match = needs_format_match(df) if (match_jpeg and real_q) else [False] * len(df)
        self.real_q = list(real_q or [])
        self.crop, self.grid, self.perturb, self.long_side = crop, grid, perturb, long_side

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        h_ = zlib.crc32(self.paths[i].encode())        # deterministic per image
        if self.recompress:
            lo, hi = self.recompress
            img = jpeg(img, lo + h_ % (hi - lo + 1))
        elif self.match[i]:
            img = jpeg(img, self.real_q[h_ % len(self.real_q)])
        if self.long_side:
            img = resize_long_side(img, self.long_side, self.crop)
        if self.perturb:
            img = self.perturb(img)
        arr = _pad_to(np.asarray(img), self.crop)
        h, w = arr.shape[:2]
        c = self.crop
        crops = [_to_tensor(arr[y:y + c, x:x + c]) for y in _grid(h, c, self.grid) for x in _grid(w, c, self.grid)]
        return torch.stack(crops), self.labels[i], 0, i


def balanced_sampler(df, num_samples, real_frac=0.5, seed=0):
    """Each draw: real with prob real_frac, otherwise a fake source chosen uniformly."""
    counts = df.cls.value_counts()
    fakes = [c for c in counts.index if c != "real"]
    w = df.cls.map(lambda c: real_frac / counts[c] if c == "real" else (1 - real_frac) / len(fakes) / counts[c])
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(w.to_numpy(dtype="float64", copy=True)), num_samples,
                                 replacement=True, generator=g)