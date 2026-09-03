"""Write a stacked TIFF test case in the Grand Challenge input layout from PNGs (or random frames if none)."""
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test/input/interface_0/images/stacked-barretts-esophagus-endoscopy")
src = [Path(p) for p in sys.argv[2:]]
out.mkdir(parents=True, exist_ok=True)
if src:
    frames = np.stack([np.array(Image.open(p).convert("RGB").resize((512, 512))) for p in src])
else:
    frames = (np.random.default_rng(0).random((6, 512, 512, 3)) * 255).astype(np.uint8)
img = sitk.GetImageFromArray(frames, isVector=True)
sitk.WriteImage(img, str(out / "stack.tif"))
print("wrote", out / "stack.tif", frames.shape)
