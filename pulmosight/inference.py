"""
PulmoSight inference: classify one chest X-ray as TB or Normal with a chosen model or the ensemble.

* Preprocessing matches prepare_cache.py (16-bit scaling -> 8-bit grayscale -> 512x512 bicubic).
* Each network's probability is temperature-calibrated (notebook 05). The ensemble is a weighted average of the three
  calibrated probabilities (notebook 11). Weights, temperatures and thresholds were all fitted on the validation set,
  never on the test set.

Usage:
    python -m pulmosight.inference path/to/xray.png [--model ensemble|efficientnet_b0|pulmonet|simple_cnn]

Research / portfolio project only. Not a medical device and not a diagnosis.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from pulmosight import models as pulmosight_models

ROOT = Path(__file__).resolve().parent.parent          # repository root
RESULTS_DIR = ROOT / "results" / "comparison"
NETWORKS = ["simple_cnn", "pulmonet", "efficientnet_b0"]

# key -> (display name, one-line description), in the order offered to users (recommended first)
MODEL_INFO = {
    "ensemble": ("Ensemble", "Weighted vote of all three networks (EfficientNet 50%, Simple CNN 30%, PulmoNet 20%)."),
    "efficientnet_b0": ("EfficientNet-B0", "Transfer learning: ImageNet-pretrained network fine-tuned on chest X-rays."),
    "pulmonet": ("PulmoNet", "Custom CNN with residual and squeeze-excite blocks, trained from scratch."),
    "simple_cnn": ("Simple CNN", "Basic CNN: five convolution + pooling blocks, trained from scratch."),
}

# 93% of the training images are stored with inverted polarity (see notebook 07). On the test set, converting uploads to
# one polarity LOWERED accuracy (EfficientNet 85.3% -> 84.1%), because the models learned both. Images are fed as uploaded.


def _read(name, notebook):
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Run notebooks/{notebook} first: results/comparison/{name} is missing.")
    return json.loads(path.read_text())


def load_results():
    """Per-network test results, calibration and thresholds (notebook 05)."""
    return _read("comparison_3models.json", "05_Model_Comparison.ipynb")


def load_ensemble():
    """Ensemble weights, thresholds and test results (notebook 11)."""
    return _read("ensemble.json", "11_Ensemble.ipynb")


def load_gray(source):
    """Open an image (path or file object) as 8-bit grayscale, exactly like prepare_cache.py.
    Also returns the original mode and how colourful the image is, so the caller can warn about photos."""
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im)
        mode, colourfulness = im.mode, 0.0
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):           # 16-bit X-rays: scale to 0-255, don't clip
            arr = np.asarray(im, dtype=np.float32)
            arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1.0) * 255
            im = Image.fromarray(arr.astype(np.uint8))
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            im = Image.alpha_composite(Image.new("RGBA", im.size, (0, 0, 0, 255)), im)
        if im.mode in ("RGB", "RGBA"):
            rgb = np.asarray(im.convert("RGB").resize((256, 256)), dtype=np.float32)
            colourfulness = float(np.abs(rgb - rgb.mean(axis=2, keepdims=True)).mean())
        return im.convert("L"), mode, colourfulness


def polarity_score(arr):
    """Centre (spine/mediastinum) minus lung-field brightness. > 0: standard polarity, < 0: inverted. (notebook 07)"""
    s = arr.shape[0]
    rows = slice(int(s * .30), int(s * .70))
    centre = arr[rows, int(s * .45):int(s * .55)].mean()
    lungs = (arr[rows, int(s * .18):int(s * .33)].mean() + arr[rows, int(s * .67):int(s * .82)].mean()) / 2
    return float(centre - lungs)


def certainty(p, thr):
    """How far the probability moved from the threshold towards 0% or 100%, relative to the room on that side."""
    relative = (p - thr) / (1 - thr) if p >= thr else (thr - p) / thr
    return "Clear" if relative >= 0.6 else "Moderate" if relative >= 0.25 else "Borderline"


class Screener:
    """Loads the three trained networks once; predict() runs a single network or the ensemble."""

    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.results, self.ensemble = load_results(), load_ensemble()
        by_arch = {m["arch"]: m for m in self.results["models"].values()}
        self.nets, self.models = {}, {}
        for arch in NETWORKS:
            ckpt = torch.load(ROOT / "models" / arch / "best.pth", map_location="cpu", weights_only=False)
            net, _ = pulmosight_models.build(ckpt["arch"])
            net.load_state_dict(ckpt["state_dict"])
            res = by_arch[arch]
            self.nets[arch] = {"net": net.to(self.device).eval(), "mean": ckpt.get("mean", 0.449), "std": ckpt.get("std", 0.226),
                               "temperature": res["temperature"]}
            self.models[arch] = {"name": MODEL_INFO[arch][0], "blurb": MODEL_INFO[arch][1], "members": {arch: 1.0},
                                 "thresholds": res["thresholds"], "test": res["metrics"]["balanced"]}
        e = self.ensemble
        self.models["ensemble"] = {"name": MODEL_INFO["ensemble"][0], "blurb": MODEL_INFO["ensemble"][1], "members": e["members"],
                                   "thresholds": {"balanced": e["threshold"], "high_spec": e["high_spec_threshold"]},
                                   "test": e["test_metrics"]}
        self.models = {k: self.models[k] for k in MODEL_INFO}              # recommended first

    @torch.no_grad()
    def _network_probability(self, arch, arr):
        n = self.nets[arch]
        x = torch.from_numpy((arr / 255.0 - n["mean"]) / n["std"]).float()[None, None].to(self.device)
        logits = n["net"](x).float()[0]
        return float(torch.sigmoid((logits[1] - logits[0]) / n["temperature"]))

    def predict(self, source, model="ensemble"):
        spec = self.models[model]
        t0 = time.perf_counter()
        gray, mode, colourfulness = load_gray(source)
        width, height = gray.size
        small = gray.resize((512, 512), Image.BICUBIC)                   # same as prepare_cache.py
        arr = np.asarray(small, dtype=np.float32)
        standard = polarity_score(arr) > 0

        votes = {arch: self._network_probability(arch, arr) for arch in spec["members"]}
        p_tb = float(sum(w * votes[arch] for arch, w in spec["members"].items()))
        thr = spec["thresholds"]["balanced"]

        warnings = []
        if min(width, height) < 300:
            warnings.append(f"Low resolution ({width}×{height}). The models were trained on full-size X-rays; results may be less reliable.")
        if colourfulness > 8:
            warnings.append("This looks like a colour image. The models expect a grayscale chest X-ray.")
        if not 0.6 <= width / height <= 1.6:
            warnings.append(f"Unusual aspect ratio ({width}×{height}). Make sure this is a full frontal chest X-ray.")
        if float(arr.std()) < 25:
            warnings.append("Very low contrast. Over- or under-exposed images are harder to classify.")

        return {
            "model_key": model, "model": spec["name"], "blurb": spec["blurb"],
            "prediction": "TB" if p_tb >= thr else "Normal", "p_tb": p_tb, "threshold": thr,
            "high_spec_threshold": spec["thresholds"]["high_spec"], "confidence": certainty(p_tb, thr), "test": spec["test"],
            "votes": {MODEL_INFO[a][0]: {"p_tb": p, "weight": spec["members"][a], "threshold": self.models[a]["thresholds"]["balanced"],
                                         "says": "TB" if p >= self.models[a]["thresholds"]["balanced"] else "Normal"}
                      for a, p in votes.items()} if model == "ensemble" else {},
            "display": small if standard else ImageOps.invert(small),        # always shown the standard way
            "original_size": (width, height), "original_mode": mode,
            "polarity": "standard (air dark)" if standard else "inverted (air bright)",
            "warnings": warnings, "seconds": time.perf_counter() - t0, "device": str(self.device),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a chest X-ray as TB or Normal.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model", default="ensemble", choices=list(MODEL_INFO))
    args = parser.parse_args()
    r = Screener().predict(args.image_path, args.model)
    print(f"{r['model']}: {r['prediction']}  P(TB) {r['p_tb']:.1%}  threshold {r['threshold']:.2f}  ({r['confidence']})")
    for name, v in r["votes"].items():
        print(f"  {name:<16} P(TB) {v['p_tb']:.1%}  weight {v['weight']:.0%}  -> {v['says']}")
    for w in r["warnings"]:
        print("  warning:", w)
