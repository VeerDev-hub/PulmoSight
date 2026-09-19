"""
Resize every X-ray ONCE to a fixed-size 8-bit grayscale PNG.

Training then reads small files instead of decoding and resizing the
full-resolution originals every epoch (the usual bottleneck on a single GPU).
Keeps the train/ and val/ class-folder layout.

Usage:
    python prepare_cache.py --src "D:/IN-CXR" --dst "D:/IN-CXR_512" --size 512
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def convert(job):
    src, dst, size = job
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        if im.mode in ('I;16', 'I'):  # 16-bit X-rays: scale to 0-255, don't clip
            arr = np.asarray(im, dtype=np.float32)
            arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1.0) * 255
            im = Image.fromarray(arr.astype(np.uint8))
        im = im.convert('L').resize((size, size), Image.BICUBIC)
        im.save(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--dst', type=Path, required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    jobs = [
        (f, (args.dst / f.relative_to(args.src)).with_suffix('.png'), args.size)
        for f in args.src.rglob('*')
        if f.suffix.lower() in EXTS
    ]
    print(f'{len(jobs)} images -> {args.dst}')
    with ProcessPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(convert, jobs, chunksize=64), total=len(jobs)))


if __name__ == '__main__':
    main()