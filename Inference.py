"""
TB screening inference + Grad-CAM heatmap.

Matches the checkpoints written by the new train.py (1-channel EfficientNet-B0,
mean/std and class names stored inside the checkpoint).

Usage:
    python Inference.py path/to/xray.png --heatmap-path outputs/heatmap.png

Research / screening-support tool only. Not a medical diagnosis.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageDraw
from torchvision import models

PROJECT_PATH = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_PATH / 'models' / 'tb_best_model.pth'
DEFAULT_THRESHOLD = 0.90


def build_model():
    model = models.efficientnet_b0(weights=None)
    stem = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, stem.out_channels, stem.kernel_size, stem.stride, stem.padding, bias=False)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 2))
    return model


def load_gray(path):
    """Same conversion as prepare_cache.py so inference sees what training saw."""
    with Image.open(path) as im:
        if im.mode in ('I;16', 'I'):
            arr = np.asarray(im, dtype=np.float32)
            arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1.0) * 255
            im = Image.fromarray(arr.astype(np.uint8))
        return im.convert('L')


def render_overlay(img, cam, box_level=0.6):
    """Create an attribution overlay and an optional broad activation box."""
    w, h = img.size
    cam_full = np.asarray(Image.fromarray(cam.astype(np.float32)).resize((w, h), Image.BILINEAR))
    cam_full = np.clip(cam_full, 0.0, 1.0)

    # Jet matches the conventional blue-green-yellow-red Grad-CAM presentation.
    heat = colormaps['jet'](cam_full)[..., :3]
    heatmap = Image.fromarray(np.uint8(heat * 255), mode='RGB')
    base = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    alpha = (0.52 * cam_full)[..., None]  # low activation -> original image shows through
    blended = (base * (1 - alpha) + heat * alpha) * 255
    overlay = Image.fromarray(blended.astype(np.uint8))

    box = None
    ys, xs = np.where(cam_full >= box_level)
    if len(xs):
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        ImageDraw.Draw(overlay).rectangle(box, outline=(0, 255, 0), width=max(2, w // 300))
    return heatmap, overlay, box


def lung_window(img):
    """Apply a conservative display window around the expected lung fields.

    This is a visualization aid, not an anatomical segmentation model.
    """
    array = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
    height, width = array.shape
    yy, xx = np.ogrid[:height, :width]
    left = (((xx - width * 0.32) / (width * 0.25)) ** 2 +
            ((yy - height * 0.53) / (height * 0.39)) ** 2) <= 1
    right = (((xx - width * 0.68) / (width * 0.25)) ** 2 +
             ((yy - height * 0.53) / (height * 0.39)) ** 2) <= 1
    mask = np.maximum(left, right).astype(np.float32)
    softened = array * (0.25 + 0.75 * mask)
    return Image.fromarray(np.uint8(np.clip(softened, 0, 1) * 255), mode='L')


def save_report(img, heatmap, overlay, result, output_path):
    """Save a reference-style original/attribution comparison report."""
    panel_width = 540
    panel_height = int(img.height * panel_width / img.width)
    lung_img = lung_window(img)
    _, lung_overlay, _ = render_overlay(lung_img, np.asarray(heatmap.convert('L')) / 255.0)
    panels = [
        [img.convert('RGB'), heatmap, overlay],
        [lung_img.convert('RGB'), heatmap, lung_overlay],
    ]
    labels = ['Original CXR', 'TB attribution heatmap', 'Overlay',
              'Lung-window visual aid', 'Attribution in lung window', 'Lung-window overlay']
    header_height = 86
    row_label_width = 150
    report = Image.new('RGB', (panel_width * 3 + row_label_width, panel_height * 2 + header_height), 'white')
    draw = ImageDraw.Draw(report)
    for row, row_panels in enumerate(panels):
        y = header_height + row * panel_height
        row_title = 'Original' if row == 0 else 'Lung-window\nvisual aid'
        draw.multiline_text((14, y + panel_height // 2 - 18), row_title, fill='#17211f', spacing=4)
        for column, panel in enumerate(row_panels):
            x = row_label_width + column * panel_width
            resized = panel.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
            report.paste(resized, (x, y))
            draw.text((x + 14, y + 14), labels[row * 3 + column], fill='white', stroke_width=2, stroke_fill='#17211f')
    draw.text((row_label_width, 14), f"Prediction: {result['prediction']}   TB probability: {result['tb_probability']}   "
                                f"Threshold: {result['threshold']:.2f}", fill='#17211f')
    draw.text((row_label_width, 43), 'Grad-CAM shows model attribution; it is not lesion localization or severity.', fill='#697773')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path)


class TBPredictor:
    """Load the model once, then call predict() as many times as you like
    (this is also what the FastAPI endpoint will use)."""

    def __init__(self, model_path=DEFAULT_MODEL, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
        self.classes = ckpt['classes']
        lowered = [c.lower() for c in self.classes]
        self.pos_idx = lowered.index(str(ckpt.get('positive_class', 'tb')).lower())
        self.size = ckpt.get('image_size', 512)
        self.mean = ckpt.get('mean', 0.449)
        self.std = ckpt.get('std', 0.226)
        self.model = build_model()
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device).eval()

    def _tensor(self, img):
        small = img.resize((self.size, self.size), Image.BICUBIC)  # same as prepare_cache.py
        arr = (np.asarray(small, dtype=np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(arr)[None, None].to(self.device)

    def _forward(self, x, with_cam):
        if not with_cam:
            with torch.no_grad():
                return self.model(x).float().softmax(1)[0], None

        store = {}

        def on_grad(grad):
            store['grad'] = grad

        def on_forward(_module, _inputs, output):
            store['act'] = output
            output.register_hook(on_grad)  # tensor hook: safe with in-place SiLU

        handle = self.model.features[-1].register_forward_hook(on_forward)
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            logits[0, self.pos_idx].backward()  # heatmap for the TB class
            weights = store['grad'].mean(dim=(2, 3), keepdim=True)
            cam = (weights * store['act']).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            return logits.detach().float().softmax(1)[0], cam.detach().cpu().numpy()
        finally:
            handle.remove()

    def predict(self, image_path, threshold=DEFAULT_THRESHOLD, heatmap_path=None):
        img = load_gray(image_path)
        probs, cam = self._forward(self._tensor(img), heatmap_path is not None)

        p_tb = probs[self.pos_idx].item()
        is_tb = p_tb >= threshold
        label = self.classes[self.pos_idx if is_tb else 1 - self.pos_idx].upper()
        result = {
            'prediction': label,
            'tb_probability': f'{p_tb * 100:.1f}%',
            'confidence': f'{(p_tb if is_tb else 1 - p_tb) * 100:.1f}%',
            'threshold': threshold,
            'severity': 'UNAVAILABLE: binary TB/Normal model has no severity labels',
            'location': 'ATTRIBUTION ONLY: not confirmed lesion localization',
        }
        if cam is not None:
            heatmap, overlay, box = render_overlay(img, cam)
            heatmap_path = Path(heatmap_path)
            heatmap_path.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(heatmap_path)
            raw_heatmap_path = heatmap_path.with_name(heatmap_path.stem + '_heatmap.png')
            original_path = heatmap_path.with_name(heatmap_path.stem + '_original.png')
            report_path = heatmap_path.with_name(heatmap_path.stem + '_report.png')
            heatmap.save(raw_heatmap_path)
            img.save(original_path)
            save_report(img, heatmap, overlay, result, report_path)
            result['heatmap_path'] = str(heatmap_path)
            result['raw_heatmap_path'] = str(raw_heatmap_path)
            result['original_path'] = str(original_path)
            result['report_path'] = str(report_path)
            result['region_box_xyxy'] = box
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify a chest X-ray for TB screening.')
    parser.add_argument('image_path', type=Path)
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--heatmap-path', type=Path, help='Save a Grad-CAM overlay to this path.')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='Provisional validation-selected TB threshold (override as needed).')
    args = parser.parse_args()
    print(TBPredictor(args.model_path).predict(args.image_path, args.threshold, args.heatmap_path))