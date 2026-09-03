# rare26-algorithm

Grand Challenge algorithm container for RARE26 (Detect Rare Early-Stage Cancers in Endoscopy, EndoVis / MICCAI 2026):
binary NDBE-vs-neoplasia classification of Barrett's esophagus endoscopy frames, scored by median PPV@90% recall
under 1:100 prevalence resampling.

- `train/train.py` — k-fold timm classifiers (weighted sampling, strong augmentation, BCE), out-of-fold logits,
  per-member affine calibration under the 1:100 prior, ensemble manifest (`resources/ensemble.json`).
- `train/score.py` — exact re-implementation of the challenge metric for local validation.
- `inference.py` — reads the stacked input image, runs the calibrated mean-logit ensemble with horizontal-flip TTA,
  writes `stacked-neoplastic-lesion-likelihoods.json`.
- `do_build.sh` / `do_test_run.sh` / `do_save.sh` — local container build, smoke test, export.
- `.github/workflows` — CI build + release of the image, upload to Grand Challenge via `gcapi` (`GC_TOKEN` secret).

Training data: `TimJaspersTue/RARE25-train` (Hugging Face, CC-BY-NC-SA-4.0, gated). Pretrained backbones from timm
(ImageNet / DINOv2) and, when available, GastroNet-5M (CC-BY-NC-4.0), all publicly licensed and disclosed.

License: MIT.
