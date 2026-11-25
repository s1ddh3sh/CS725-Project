"""Tri-branch forensic network with hierarchical attribution heads.

  Branch A: CLIP ViT (multi-layer CLS taps, LoRA)      -> 512-d token
  Branch B: SRM / learnable high-pass -> ResNet        -> 512-d token
  Branch C: FFT log-spectrum of residual -> small CNN  -> 512-d token
  Fusion  : 2-layer Transformer over [CLS, A, B, C] -> 256-d embedding
  Heads   : N-way ArcFace cosine, binary, family (aux)
            (binary_only=True keeps just the real/fake head; used by the COCO scripts)

num_classes / num_families default to the Synthbuster setup in classes.py.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from classes import NUM_CLASSES, NUM_FAMILIES

D_TOK = 512   # token width of every branch
D_EMB = 256   # embedding width


# =====================================================================  Branch A
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.lora_a = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


class CLIPBranch(nn.Module):
    def __init__(self, name="openai/clip-vit-large-patch14", lora_r=8, lora_last=8,
                 tap_fracs=(1 / 3, 1 / 2, 2 / 3, 5 / 6, 1.0)):
        super().__init__()
        from transformers import CLIPVisionModel
        self.clip = CLIPVisionModel.from_pretrained(name)
        for p in self.clip.parameters():
            p.requires_grad = False
        n, hid = self.clip.config.num_hidden_layers, self.clip.config.hidden_size
        # transformers 4.x nests the encoder under .vision_model; 5.x puts it on the model itself
        vm = getattr(self.clip, "vision_model", self.clip)
        for layer in vm.encoder.layers[-lora_last:]:
            layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, lora_r, 2 * lora_r)
            layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, lora_r, 2 * lora_r)
        # hidden_states[i] = output of block i (i = 1..n); index 0 is the embedding output
        self.taps = sorted({max(1, round(f * n)) for f in tap_fracs})
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in self.taps])
        self.proj = nn.ModuleList([nn.Linear(hid, D_TOK) for _ in self.taps])
        self.tap_logits = nn.Parameter(torch.zeros(len(self.taps)))
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(self, x01):
        x = (x01 - self.mean) / self.std
        hs = self.clip(pixel_values=x, output_hidden_states=True).hidden_states
        feats = [p(nm(hs[t][:, 0])) for t, nm, p in zip(self.taps, self.norms, self.proj)]
        w = torch.softmax(self.tap_logits, 0)
        return sum(wi * f for wi, f in zip(w, feats))


# =====================================================================  Branch B
def _srm_weight():
    k1 = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], np.float32) / 4.0
    k2 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12.0
    k3 = np.zeros((5, 5), np.float32)
    k3[2, 1:4] = [1, -2, 1]
    k3 = k3 / 2.0
    k4 = k3.T.copy()                     # vertical 2nd derivative
    ks = np.stack([k1, k2, k3, k4])      # [4,5,5]
    w = np.tile(ks, (3, 1, 1))[:, None]  # [12,1,5,5]; out ch 4*c+j = kernel j on RGB channel c
    return torch.from_numpy(w)


class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(True), nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean((2, 3)))[:, :, None, None]


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, se=False):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.se = SE(cout) if se else nn.Identity()
        self.short = (nn.Identity() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)), True)
        o = self.se(self.b2(self.c2(o)))
        return F.relu(o + self.short(x), True)


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(1.0, 6.0)
        return x.clamp(min=self.eps).pow(p).mean((-2, -1)).pow(1.0 / p)


class ForensicBranch(nn.Module):
    """Fixed SRM residuals (12 ch) + learnable high-pass (3 ch) -> ResNet without early downsampling."""

    def __init__(self, trunc=3.0):
        super().__init__()
        self.register_buffer("srm", _srm_weight())
        self.trunc = trunc
        self.hp = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        with torch.no_grad():
            self.hp.weight.zero_()
            lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32) / 4
            for c in range(3):
                self.hp.weight[c, c] = lap
        self.in_bn = nn.BatchNorm2d(15)
        self.stem = nn.Sequential(nn.Conv2d(15, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        cfg = [(64, 64, 1, 2, False), (64, 128, 2, 2, False), (128, 256, 2, 3, True), (256, 512, 2, 2, True)]
        layers = []
        for cin, cout, stride, n, se in cfg:
            layers.append(BasicBlock(cin, cout, stride, se))
            layers += [BasicBlock(cout, cout, 1, se) for _ in range(n - 1)]
        self.body = nn.Sequential(*layers)
        self.pool = GeM()
        self.out = nn.Sequential(nn.Linear(512, D_TOK), nn.LayerNorm(D_TOK))

    def forward(self, x01):
        with torch.autocast(device_type=x01.device.type, enabled=False):  # keep residuals in fp32
            x255 = x01.float() * 255.0
            srm = F.conv2d(x255, self.srm, padding=2, groups=3).clamp(-self.trunc, self.trunc) / self.trunc
            hp = self.hp(x255) / 16.0
            feat = torch.cat([srm, hp], 1)
        h = self.stem(self.in_bn(feat))
        return self.out(self.pool(self.body(h)))


# =====================================================================  Branch C
class SpectrumBranch(nn.Module):
    def __init__(self, crop=224):
        super().__init__()
        k = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32) / 8
        self.register_buffer("hp", k.view(1, 1, 3, 3))
        win = torch.hann_window(crop, periodic=False)
        self.register_buffer("win", (win[:, None] * win[None, :]).view(1, 1, crop, crop))

        def blk(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
                                 nn.MaxPool2d(2))
        self.cnn = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        self.out = nn.Sequential(nn.Linear(256, D_TOK), nn.LayerNorm(D_TOK))

    def forward(self, x01):
        with torch.autocast(device_type=x01.device.type, enabled=False):
            x = x01.float()
            g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]) * 255.0
            r = F.conv2d(g, self.hp, padding=1) * self.win
            spec = torch.fft.fftshift(torch.fft.fft2(r), dim=(-2, -1))
            s = torch.log1p(spec.abs())
            s = (s - s.mean((-2, -1), keepdim=True)) / (s.std((-2, -1), keepdim=True) + 1e-6)
        return self.out(self.cnn(s).mean((2, 3)))


# =====================================================================  Heads
class ArcMarginHead(nn.Module):
    """Cosine classifier with additive angular margin (ArcFace)."""

    def __init__(self, d, n, s=30.0, m=0.2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n, d))
        nn.init.xavier_uniform_(self.W)
        self.n, self.s, self.m = n, s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m

    def forward(self, z, y=None):
        cos = F.linear(F.normalize(z.float()), F.normalize(self.W.float())).clamp(-1 + 1e-7, 1 - 1e-7)
        if y is None:
            return cos * self.s
        sin = torch.sqrt((1 - cos ** 2).clamp(min=1e-6))
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        onehot = F.one_hot(y, self.n).bool()
        return torch.where(onehot, phi, cos) * self.s


# =====================================================================  Full model
class AttributionNet(nn.Module):
    def __init__(self, branches="abc", clip_name="openai/clip-vit-large-patch14", crop=224,
                 lora_r=8, lora_last=8, branch_drop=0.15, arc_s=30.0, arc_m=0.2, fusion_drop=0.2,
                 num_classes=NUM_CLASSES, num_families=NUM_FAMILIES, binary_only=False):
        super().__init__()
        self.binary_only = binary_only      # True: only the real/fake head exists (no attribution heads)
        self.branches = [b for b in "abc" if b in branches]
        assert self.branches, "need at least one branch"
        self.branch_drop = branch_drop
        if "a" in self.branches:
            self.branch_a = CLIPBranch(clip_name, lora_r, lora_last)
        if "b" in self.branches:
            self.branch_b = ForensicBranch()
        if "c" in self.branches:
            self.branch_c = SpectrumBranch(crop)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, D_TOK))
        self.type_emb = nn.Parameter(torch.randn(1, 4, D_TOK) * 0.02)  # 0=cls, 1..3 = a,b,c
        layer = nn.TransformerEncoderLayer(D_TOK, 8, 1024, fusion_drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_TOK)
        self.embed = nn.Linear(D_TOK, D_EMB)
        self.bin_head = nn.Linear(D_EMB, 1)
        if not binary_only:
            self.cls_head = ArcMarginHead(D_EMB, num_classes, arc_s, arc_m)
            self.fam_head = nn.Linear(D_EMB, num_families)

    def set_stage(self, stage):
        """stage 1: CLIP fully frozen. stage 2: LoRA weights trainable."""
        for n, p in self.named_parameters():
            if "lora_" in n:
                p.requires_grad = stage == 2
            elif n.startswith("branch_a.clip."):
                p.requires_grad = False

    def forward(self, x01, y=None):
        toks = []
        for b in self.branches:
            toks.append(getattr(self, f"branch_{b}")(x01))
        t = torch.stack(toks, 1)                                   # [B, nb, D]
        if self.training and self.branch_drop > 0 and len(self.branches) > 1:
            B, nb = t.shape[:2]
            drop = torch.rand(B, device=t.device) < self.branch_drop
            which = torch.randint(0, nb, (B,), device=t.device)
            mask = torch.zeros(B, nb, device=t.device)
            mask[torch.arange(B, device=t.device), which] = drop.float()
            t = t * (1 - mask).unsqueeze(-1)
        idx = torch.tensor([0] + [1 + "abc".index(b) for b in self.branches], device=t.device)
        t = torch.cat([self.cls_tok.expand(len(t), -1, -1), t], 1) + self.type_emb[:, idx]
        h = self.embed(self.norm(self.fusion(t)[:, 0]))            # pre-norm embedding [B, 256]
        z = F.normalize(h, dim=-1)
        out = dict(z=z, h=h, bin=self.bin_head(h).squeeze(-1))
        if self.binary_only:
            return out
        out["logits"] = self.cls_head(z)
        out["fam"] = self.fam_head(h)
        if y is not None:
            out["logits_margin"] = self.cls_head(z, y)
        return out