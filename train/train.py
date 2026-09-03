"""RARE26 training: k-fold timm classifiers on center_*/{ndbe,neo}/*.png, OOF logits, affine calibration,
ensemble manifest for inference.py. Runs on Colab/Kaggle GPU, MPS, or CPU (synthetic smoke test).

  python train/train.py --data data/train --out resources --arch resnet50.a1_in1k --img 224 --epochs 30
  python train/train.py --data data/train --out resources --arch convnext_base.fb_in22k_ft_in1k --img 224 --epochs 25
  python train/train.py --data data/train --out resources --finalize     # calibrate + write ensemble.json + OOF score

Each run appends its folds to resources/ensemble.json; --finalize re-fits calibration on every member's OOF
logits and reports the bootstrap PPV@90 of the calibrated mean-logit ensemble on the OOF set.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import bootstrap_metrics, ppv_at_recall  # noqa: E402

IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def list_images(root: Path):
    rows = []
    for center_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for label_name, label in (("ndbe", 0), ("neo", 1)):
            for f in sorted((center_dir / label_name).glob("*.png")):
                rows.append((str(f), label, center_dir.name))
    if not rows:
        raise SystemExit(f"no images under {root} (expected center_*/{{ndbe,neo}}/*.png)")
    return rows


class ImgDS(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label, _ = self.rows[i]
        img = self._cache.get(path)
        if img is None:
            img = Image.open(path).convert("RGB")
            img.load()
            self._cache[path] = img
        return self.tf(img), torch.tensor(float(label))


def make_tf(img: int, train: bool):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation(20)], p=0.5),
            T.ColorJitter(0.25, 0.25, 0.2, 0.03),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
            T.ToTensor(), T.Normalize(**IMAGENET), T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    return T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(**IMAGENET)])


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device, tta=True):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device)
        z = model(x).squeeze(1).float()
        if tta:
            z = 0.5 * (z + model(torch.flip(x, dims=[3])).squeeze(1).float())
        out.append(z.cpu())
    return torch.cat(out).numpy()


def train_fold(args, rows, tr_idx, va_idx, fold, device, tag):
    seed = args.seed + fold
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    tr_rows = [rows[i] for i in tr_idx]; va_rows = [rows[i] for i in va_idx]
    y_tr = np.array([r[1] for r in tr_rows]); y_va = np.array([r[1] for r in va_rows])
    # weighted sampler: positives drawn to ~args.pos_frac of every batch
    w_pos = args.pos_frac / max(y_tr.sum(), 1); w_neg = (1 - args.pos_frac) / max((y_tr == 0).sum(), 1)
    weights = np.where(y_tr == 1, w_pos, w_neg)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(tr_rows), replacement=True)
    nw = 0 if device.type == "mps" else args.workers
    tr_dl = DataLoader(ImgDS(tr_rows, make_tf(args.img, True)), batch_size=args.bs, sampler=sampler, num_workers=nw, drop_last=True, persistent_workers=nw > 0)
    va_dl = DataLoader(ImgDS(va_rows, make_tf(args.img, False)), batch_size=args.bs * 2, shuffle=False, num_workers=nw)

    model = timm.create_model(args.arch, pretrained=not args.no_pretrained, num_classes=1, drop_rate=args.drop)
    if args.init_weights:
        sd = torch.load(args.init_weights, map_location="cpu")
        sd = sd.get("state_dict", sd.get("teacher", sd))
        sd = {k.replace("backbone.", "").replace("module.", ""): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not k.startswith(("fc.", "head.", "classifier."))}
        print("init_weights:", model.load_state_dict(sd, strict=False))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(tr_dl); warm = min(len(tr_dl), steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best = (-1.0, None, None)
    step = 0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                z = model(x).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(z.float(), y * (1 - args.smooth) + 0.5 * args.smooth)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            tot += loss.item()
        z_va = predict(model, va_dl, device, tta=False)
        ppv = ppv_at_recall(y_va, z_va); auc_proxy = float(np.mean(z_va[y_va == 1][:, None] > z_va[y_va == 0][None, :])) if y_va.sum() else 0.0
        print(f"[{tag} f{fold}] ep{ep + 1}/{args.epochs} loss={tot / len(tr_dl):.4f} val_ppv@90={ppv:.4f} val_auc={auc_proxy:.4f} {time.time() - t0:.0f}s", flush=True)
        # select on ppv@90 with AUC as tie-break; skip the first 20% of epochs (noisy)
        key = ppv + 1e-3 * auc_proxy
        if ep + 1 >= max(1, args.epochs // 5) and key > best[0]:
            best = (key, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    z_va = predict(model, va_dl, device, tta=True)
    print(f"[{tag} f{fold}] best epoch {best[2]} TTA val_ppv@90={ppv_at_recall(y_va, z_va):.4f}")
    return best[1], z_va, best[2]


def fit_affine(z: np.ndarray, y: np.ndarray, prior_pos: float = 1 / 101, iters: int = 500):
    """Affine logit calibration (t, b) minimising prior-reweighted log loss: weights classes so the fit reflects
    1:100 deployment prevalence (IMSY's recipe). Returns (t, b)."""
    zt = torch.tensor(z, dtype=torch.float64); yt = torch.tensor(y, dtype=torch.float64)
    w = torch.where(yt == 1, prior_pos / max(y.mean(), 1e-9), (1 - prior_pos) / max(1 - y.mean(), 1e-9))
    t = torch.ones(1, dtype=torch.float64, requires_grad=True); b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([t, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (w * F.binary_cross_entropy_with_logits(t * zt + b, yt, reduction="none")).sum() / w.sum()
        loss.backward(); return loss
    opt.step(closure)
    return float(t.item()), float(b.item())


def finalize(args, rows):
    out = Path(args.out); manifest = json.loads((out / "ensemble.json").read_text())
    y = np.array([r[1] for r in rows]); paths = [r[0] for r in rows]
    oof_sum = np.zeros(len(rows)); n_models = {}
    for m in manifest["members"]:
        oof = np.load(out / m["oof"], allow_pickle=True).item()
        idx = np.array([paths.index(p) for p in oof["paths"]]); z = oof["logits"]
        t, b = fit_affine(z, y[idx]); m["cal_t"], m["cal_b"] = t, b
        key = m["run"]; n_models.setdefault(key, np.zeros(len(rows)))
        n_models[key][idx] += t * z + b
    # every run covers each image exactly once (its OOF fold) → mean over runs is the ensemble OOF logit
    ens = np.mean(list(n_models.values()), axis=0)
    per_run = {k: bootstrap_metrics(y, v, n_iterations=args.boot)["PPV@90RECALL"] for k, v in n_models.items()}
    res = bootstrap_metrics(y, ens, n_iterations=args.boot)
    centers = np.array([r[2] for r in rows])
    res["per_center"] = {c: bootstrap_metrics(y[centers == c], ens[centers == c], n_iterations=args.boot)["PPV@90RECALL"] for c in sorted(set(centers))}
    res["per_run"] = per_run
    manifest["oof_score"] = res
    (out / "ensemble.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="resources")
    ap.add_argument("--arch", default="resnet50.a1_in1k"); ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--drop", type=float, default=0.2); ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--pos_frac", type=float, default=0.25); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_folds", type=str, default=""); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--run", default="")
    ap.add_argument("--init_weights", default=""); ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--finalize", action="store_true"); ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    rows = list_images(Path(args.data)); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    y = np.array([r[1] for r in rows]); print(f"{len(rows)} images, {int(y.sum())} neo, centers={sorted(set(r[2] for r in rows))}")
    if args.finalize:
        return finalize(args, rows)
    device = pick_device(); print("device", device)
    run = args.run or f"{args.arch.split('.')[0]}_{args.img}_s{args.seed}"
    strat = np.array([f"{r[2]}_{r[1]}" for r in rows])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    manifest_p = out / "ensemble.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {"members": [], "tta_hflip": True}
    manifest["members"] = [m for m in manifest["members"] if m["run"] != run]
    only = {int(f) for f in args.only_folds.split(",") if f}
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(rows)), strat)):
        if only and fold not in only:
            continue
        sd, z_va, best_ep = train_fold(args, rows, tr, va, fold, device, run)
        wname = f"{run}_f{fold}.pt"; oname = f"{run}_f{fold}_oof.npy"
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in sd.items()}, out / wname)
        np.save(out / oname, {"paths": [rows[i][0] for i in va], "logits": z_va, "labels": y[va]})
        manifest["members"].append({"run": run, "fold": fold, "arch": args.arch, "img_size": args.img, "weights": wname, "oof": oname, "best_epoch": best_ep, "weight": 1.0})
        manifest_p.write_text(json.dumps(manifest, indent=2))
    print("done", run)


if __name__ == "__main__":
    main()
