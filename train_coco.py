"""Train the tri-branch network on COCO real images + DRCT-2M fakes as a pure real/fake detector.

ONE objective: binary cross-entropy (fake = 1, real = 0). There are no generator / family heads.
The generator name of each fake is still used, but only for
  * balanced sampling (50% real, 50% fake split evenly over the fake sources), and
  * per-generator reporting (TPR / AUC per source) - never as a training target.

Same two-stage recipe as the Synthbuster train.py:
  stage 1: CLIP frozen; branches B, C, fusion and the binary head train.
  stage 2: LoRA on CLIP enabled (low LR), everything else at a lower LR.
The checkpoint is chosen by validation log-loss (sensitive to calibration, unlike AUC).

  python train_coco.py --manifest /data/coco_manifest.csv --out_dir runs/coco_v1
  python train_coco.py ... --branches bc                 # ablation: no CLIP branch
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from coco_dataset import EvalDataset, TrainDataset, balanced_sampler
from coco_engine import detection_report, predict
from model import AttributionNet
from utils import EMA, cosine_warmup, load_trainable, seed_everything, trainable_state_dict


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="CSV from build_coco_manifest.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--branches", default="abc", help="subset of 'abc' for ablations")
    p.add_argument("--clip_name", default="openai/clip-vit-large-patch14")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_last", type=int, default=8)
    p.add_argument("--branch_drop", type=float, default=0.15)
    p.add_argument("--crop", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--samples_per_epoch", type=int, default=60000)
    p.add_argument("--epochs1", type=int, default=8)
    p.add_argument("--epochs2", type=int, default=8)
    p.add_argument("--lr1", type=float, default=2e-4)
    p.add_argument("--lr2", type=float, default=1e-4)
    p.add_argument("--lr2_lora", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="e.g. 0.05 -> targets 0.025 / 0.975 (can improve calibration)")
    # augmentation (identical for both classes)
    p.add_argument("--recompress", type=int, nargs=2, default=[60, 70], metavar=("LO", "HI"),
                   help="re-encode EVERY image (real and fake) as JPEG at a random quality in [LO, HI] "
                        "before anything else, so both classes end with the same JPEG tables. Keep HI below "
                        "the quality of both originals (DRCT fakes q75, COCO q~96). '0 0' disables it.")
    p.add_argument("--no_format_match", action="store_true",
                   help="(only used when --recompress is off) disable JPEG format matching of PNG fakes")
    p.add_argument("--p_rescale", type=float, default=0.5)
    p.add_argument("--scale_lo", type=float, default=0.6)
    p.add_argument("--scale_hi", type=float, default=1.4)
    p.add_argument("--p_blur", type=float, default=0.2)
    p.add_argument("--p_jpeg", type=float, default=0.3)
    p.add_argument("--val_grid", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", default=None, help="best.pt (from this binary-only script) to initialise from")
    p.add_argument("--max_bad_steps", type=int, default=25)
    return p.parse_args(argv)


def bce_loss(logit, y_fake, smoothing=0.0):
    target = y_fake.float() * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logit.float(), target)


def load_manifest(path):
    df = pd.read_csv(path)
    meta = json.load(open(Path(path).with_suffix(".meta.json")))
    df["label"] = (df.cls != "real").astype(int)          # the only target: 1 = fake, 0 = real
    return df, meta


def audit_jpeg(df, recompress):
    """Warn if the stored JPEG qualities differ between real and fake and nothing equalises them."""
    if "jpeg_q" not in df:
        return
    med = df.groupby("cls").jpeg_q.median()
    print("stored JPEG quality (median) per source:", med.dropna().astype(int).to_dict())
    if recompress:
        top = med.dropna().min()
        print(f"re-encoding every image at q in [{recompress[0]}, {recompress[1]}] (both classes)")
        if recompress[1] >= top:
            print(f"WARNING: --recompress HI ({recompress[1]}) >= lowest stored quality ({top:.0f}). Images "
                  f"stored at that quality barely change while the others are re-quantised -> keep HI lower.")
    elif med.dropna().nunique() > 1:
        print("WARNING: real and fake JPEG qualities differ and --recompress is off: the model can "
              "separate the classes by JPEG quality alone.")


def build_model(a):
    return AttributionNet(branches=a.branches, clip_name=a.clip_name, crop=a.crop, lora_r=a.lora_r,
                          lora_last=a.lora_last, branch_drop=a.branch_drop, binary_only=True)


def make_optimizer(model, lr_new, lr_lora, wd):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lora = "lora_" in n
        decay = (p.ndim > 1) and not lora
        groups.setdefault((lora, decay), []).append(p)
    pg = [dict(params=ps, lr=lr_lora if lora else lr_new, weight_decay=wd if decay else 0.0)
          for (lora, decay), ps in groups.items()]
    return torch.optim.AdamW(pg, betas=(0.9, 0.999))


def main():
    a = get_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found - training the CLIP branch on CPU is not practical.")

    df, meta = load_manifest(a.manifest)
    sources = sorted(set(df.cls) - {"real"})
    cfg = dict(vars(a), binary_only=True, sources=sources, real_jpeg_q=meta["real_jpeg_q"])
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    sp = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    for k, v in sp.items():
        v.to_csv(out / f"split_{k}.csv", index=False)
    print("fake sources:", sources)
    print(df.pivot_table(index="cls", columns="split", values="path", aggfunc="count").fillna(0).astype(int))
    # leakage guard: no COCO id may appear in two splits
    ids = {s: set(v.coco_id) for s, v in sp.items()}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]), "id leakage"

    real_q = None if a.no_format_match else meta["real_jpeg_q"]
    recompress = tuple(a.recompress) if a.recompress[1] > 0 else None
    audit_jpeg(df, recompress)
    tr_ds = TrainDataset(sp["train"], a.crop, aug=True, real_q=real_q, recompress=recompress, p_rescale=a.p_rescale,
                         scale=(a.scale_lo, a.scale_hi), p_blur=a.p_blur, p_jpeg=a.p_jpeg)
    sampler = balanced_sampler(sp["train"], a.samples_per_epoch, seed=a.seed)
    tr_dl = DataLoader(tr_ds, a.batch_size, sampler=sampler, drop_last=True, num_workers=a.workers,
                       pin_memory=True, persistent_workers=a.workers > 0)
    va_dl = DataLoader(EvalDataset(sp["val"], a.crop, a.val_grid, real_q=real_q, recompress=recompress), 16,
                       shuffle=False,
                       num_workers=a.workers, pin_memory=True)

    model = build_model(a).to(device)
    if a.resume:
        load_trainable(model, torch.load(a.resume, map_location=device)["model"])
        print(f"initialised weights from {a.resume}")
    use_amp = device.type == "cuda"
    best, log = float("inf"), []

    for stage, epochs in ((1, a.epochs1), (2, a.epochs2)):
        if epochs == 0 or (stage == 2 and "a" not in a.branches):
            continue                                   # LoRA stage only exists with the CLIP branch
        if stage == 2 and (out / "best.pt").exists():  # continue from the best stage-1 weights
            load_trainable(model, torch.load(out / "best.pt", map_location=device)["model"])
        model.set_stage(stage)
        opt = make_optimizer(model, a.lr1 if stage == 1 else a.lr2, a.lr2_lora, a.wd)
        base_lrs = [g["lr"] for g in opt.param_groups]
        total = epochs * len(tr_dl)
        warm = max(1, int(0.03 * total))
        ema = EMA(model, a.ema)
        step = 0
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n=== stage {stage}: {epochs} epochs x {len(tr_dl)} steps, trainable params {n_tr/1e6:.2f}M ===")

        for ep in range(epochs):
            model.train()
            t0, run, run_acc, skipped, bad_streak = time.time(), 0.0, 0.0, 0, 0
            pbar = tqdm(tr_dl, desc=f"s{stage} ep{ep+1}/{epochs}", leave=False)
            for i, (x, y) in enumerate(pbar):
                x, y = x.to(device, non_blocking=True), y.to(device)
                f = cosine_warmup(step, total, warm)
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * f
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logit = model(x)["bin"]
                loss = bce_loss(logit, y, a.label_smoothing)
                opt.zero_grad(set_to_none=True)
                ok = bool(torch.isfinite(loss))
                if ok:
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    ok = bool(torch.isfinite(gn))
                if not ok:                                   # skip this step, never apply bad gradients
                    opt.zero_grad(set_to_none=True)
                    skipped += 1
                    bad_streak += 1
                    step += 1
                    if bad_streak >= a.max_bad_steps:
                        raise RuntimeError(f"{bad_streak} consecutive non-finite steps. "
                                           f"Resume from best.pt with --resume and a lower --lr1.")
                    continue
                bad_streak = 0
                opt.step()
                ema.update(model)
                step += 1
                acc = ((logit.detach() > 0).long() == y).float().mean().item()
                run = 0.98 * run + 0.02 * loss.item() if i else loss.item()
                run_acc = 0.98 * run_acc + 0.02 * acc if i else acc
                if i % 20 == 0:
                    pbar.set_postfix(bce=f"{run:.3f}", acc=f"{run_acc:.3f}")

            ema.apply(model)                                     # evaluate with EMA weights
            res = predict(model, va_dl, device)
            rep = detection_report(res, sp["val"].cls.values)
            m = rep["overall"]
            rec = dict(stage=stage, epoch=ep + 1, train_bce=run, train_acc=run_acc, time=time.time() - t0,
                       skipped=skipped, **{f"val_{k}": v for k, v in m.items()},
                       **{f"val_tpr_{g}": rep[g]["tpr"] for g in sources})
            log.append(rec)
            if skipped:
                print(f"  (skipped {skipped} non-finite steps this epoch)")
            per_src = "  ".join(f"{g} TPR {rep[g]['tpr']:.3f}" for g in sources)
            print(f"[s{stage} ep{ep+1}] train bce {run:.3f} acc {run_acc:.3f} | val logloss {m['logloss']:.4f} "
                  f"AUC {m['auc']:.4f} bAcc {m['bacc']:.4f} TNR(real) {m['tnr']:.3f} | {per_src}")
            if m["logloss"] < best:
                best = m["logloss"]
                torch.save(dict(model=trainable_state_dict(model), val=rec, stage=stage, epoch=ep + 1),
                           out / "best.pt")
                print(f"  -> saved best.pt (val logloss {best:.4f})")
            ema.restore(model)
            json.dump(log, open(out / "train_log.json", "w"), indent=2)

    print(f"\nDone. Best val logloss = {best:.4f}.\nNext: python evaluate_coco.py --run_dir {out} "
          f"--synthbuster_root /path/to/synthbuster")


if __name__ == "__main__":
    main()