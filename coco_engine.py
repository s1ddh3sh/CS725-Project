"""Inference + metrics for the COCO setup (binary real/fake detection only)."""
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


@torch.no_grad()
def predict(model, loader, device):
    """Averages the real/fake logit over the K crops of every image."""
    model.eval()
    out = defaultdict(list)
    use_amp = device.type == "cuda"
    for x, y, _, _ in loader:
        B, K = x.shape[:2]
        x = x.to(device, non_blocking=True).flatten(0, 1)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            o = model(x)
        out["bin"].append(o["bin"].float().view(B, K).mean(1).cpu())
        out["y"].append(torch.as_tensor(y))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def _safe(fn, *a):
    try:
        return float(fn(*a))
    except ValueError:
        return float("nan")


def binary_metrics(y_fake, p_fake, thr=0.5):
    y_fake = np.asarray(y_fake).astype(int)
    p = np.clip(np.asarray(p_fake, dtype=np.float64), 1e-7, 1 - 1e-7)
    pred = (p > thr).astype(int)
    tpr = float(pred[y_fake == 1].mean()) if (y_fake == 1).any() else float("nan")
    tnr = float(1 - pred[y_fake == 0].mean()) if (y_fake == 0).any() else float("nan")
    return dict(auc=_safe(roc_auc_score, y_fake, p), ap=_safe(average_precision_score, y_fake, p),
                bacc=float(np.nanmean([tpr, tnr])), tpr=tpr, tnr=tnr,
                logloss=float(-np.mean(y_fake * np.log(p) + (1 - y_fake) * np.log(1 - p))))


def fake_prob(res):
    return 1 / (1 + np.exp(-res["bin"]))


def detection_report(res, groups, thr=0.5):
    """groups[i] = source name of sample i ('real' = negative). Per-source metrics are
    computed against ALL real images, so every generator gets its own AUC / TPR."""
    groups = np.asarray(groups)
    y = (groups != "real").astype(int)
    p = fake_prob(res)
    rep = {"overall": binary_metrics(y, p, thr)}
    real = groups == "real"
    for g in sorted(set(groups) - {"real"}):
        m = real | (groups == g)
        rep[g] = binary_metrics(y[m], p[m], thr)
    return rep


def print_report(title, rep):
    print(f"\n{title}")
    print(f"  {'source':16s} {'AUC':>7s} {'AP':>7s} {'bAcc':>7s} {'TPR':>7s} {'TNR':>7s} {'logloss':>8s}")
    for k, m in rep.items():
        print(f"  {k:16s} {m['auc']:7.4f} {m['ap']:7.4f} {m['bacc']:7.4f} {m['tpr']:7.4f} "
              f"{m['tnr']:7.4f} {m['logloss']:8.4f}")