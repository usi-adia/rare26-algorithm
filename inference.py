"""RARE26 inference container: stacked endoscopy image in, JSON list of neoplasia likelihoods out.

Interface (Grand Challenge socket `stacked-barretts-esophagus-endoscopy-images`):
  /input/images/stacked-barretts-esophagus-endoscopy/*.{tif,tiff,mha}  -> SimpleITK array (Z,H,W,C) or (H,W,C)
  /output/stacked-neoplastic-lesion-likelihoods.json                   -> [p_0, p_1, ...] one float per frame

Ensemble: every member in resources/ensemble.json is a timm classifier with a 1-logit head.
Each member's logit is affine-calibrated (t*z + b, fitted on its out-of-fold predictions), members are averaged in
logit space with optional hflip TTA, and a sigmoid gives the likelihood. Metric is rank-based so the sigmoid is
only cosmetic; the calibration is what makes members commensurable.
"""
from __future__ import annotations

import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK
import timm
import torch
import torch.nn.functional as F

INPUT_PATH = Path(os.environ.get("RARE_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("RARE_OUTPUT", "/output"))
RESOURCE_PATH = Path(os.environ.get("RARE_RESOURCES", Path(__file__).resolve().parent / "resources"))
BATCH = 32
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_stack(location: Path) -> np.ndarray:
    files = sorted(glob(str(location / "*.tif")) + glob(str(location / "*.tiff")) + glob(str(location / "*.mha")))
    if not files:
        raise FileNotFoundError(f"no input image under {location}")
    arr = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(files[0]))
    print(f"input {files[0]} shape={arr.shape} dtype={arr.dtype}")
    if arr.ndim == 2:              # single grayscale frame
        arr = arr[None, ..., None]
    elif arr.ndim == 3:
        # (H,W,C) single RGB frame vs (Z,H,W) grayscale stack: a colour axis is 3 or 4 wide, a stack axis rarely is.
        arr = arr[None] if arr.shape[-1] in (3, 4) else arr[..., None]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        hi = float(arr.max()) if arr.size else 1.0
        arr = (arr.astype(np.float32) * (255.0 / hi if hi > 255 else 1.0)).clip(0, 255).astype(np.uint8)
    return arr  # (Z,H,W,3) uint8


def preprocess(frames: np.ndarray, size: int) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def build_member(spec: dict, device: torch.device):
    model = timm.create_model(spec["arch"], pretrained=False, num_classes=1)
    state = torch.load(RESOURCE_PATH / spec["weights"], map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().half() if device.type == "cuda" else model.to(device).eval()


@torch.no_grad()
def member_logits(model, x: torch.Tensor, device: torch.device, tta: bool) -> np.ndarray:
    out = []
    for i in range(0, len(x), BATCH):
        xb = x[i:i + BATCH].to(device)
        if device.type == "cuda":
            xb = xb.half()
        z = model(xb).float().squeeze(1)
        if tta:
            z = 0.5 * (z + model(torch.flip(xb, dims=[3])).float().squeeze(1))
        out.append(z.cpu())
    return torch.cat(out).numpy()


def run() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__} timm={timm.__version__}")
    spec = json.loads((RESOURCE_PATH / "ensemble.json").read_text())
    frames = load_stack(INPUT_PATH / "images/stacked-barretts-esophagus-endoscopy")
    cache: dict[int, torch.Tensor] = {}
    total = np.zeros(len(frames), dtype=np.float64)
    weight_sum = 0.0
    for m in spec["members"]:
        size = int(m.get("img_size", 224))
        if size not in cache:
            cache[size] = preprocess(frames, size)
        model = build_member(m, device)
        z = member_logits(model, cache[size], device, tta=bool(spec.get("tta_hflip", True)))
        z = float(m.get("cal_t", 1.0)) * z + float(m.get("cal_b", 0.0))
        w = float(m.get("weight", 1.0))
        total += w * z
        weight_sum += w
        del model
        print(f"member {m['weights']}: mean logit {z.mean():+.3f}")
    probs = 1.0 / (1.0 + np.exp(-(total / max(weight_sum, 1e-9))))
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json").write_text(
        json.dumps([float(p) for p in probs], indent=4))
    print(f"wrote {len(probs)} likelihoods; mean={probs.mean():.4f} max={probs.max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
