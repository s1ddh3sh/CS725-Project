import io
import random

import numpy as np
import torch
from PIL import Image, ImageFilter


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- perturbations
def make_perturb(kind, param):
    """Returns a PIL->PIL function used for robustness evaluation."""
    if kind == "jpeg":
        def f(img):
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=int(param))
            buf.seek(0)
            return Image.open(buf).convert("RGB")
    elif kind == "resize":
        def f(img):
            w, h = img.size
            return img.resize((max(8, int(w * param)), max(8, int(h * param))), Image.BILINEAR)
    elif kind == "blur":
        def f(img):
            return img.filter(ImageFilter.GaussianBlur(param))
    else:
        raise ValueError(kind)
    return f


PERTURBATIONS = [("clean", None), ("jpeg", 95), ("jpeg", 85), ("jpeg", 75),
                 ("resize", 0.75), ("resize", 0.5), ("blur", 1.0)]


# ---------------------------------------------------------------- EMA
class EMA:
    """EMA over trainable parameters only (buffers such as BN stats stay live)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.steps = 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.copy_(self.backup[n])
        self.backup = {}


# ---------------------------------------------------------------- checkpoints
def trainable_state_dict(model):
    """Everything except the frozen CLIP base weights (LoRA weights are kept)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not (k.startswith("branch_a.clip.") and "lora_" not in k)}


def load_trainable(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not (k.startswith("branch_a.clip.") and "lora_" not in k)]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint mismatch. missing={bad[:5]} unexpected={list(unexpected)[:5]}")


# ---------------------------------------------------------------- schedule
def cosine_warmup(step, total, warmup, floor=0.01):
    import math
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))