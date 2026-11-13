"""Render assets/trailer/trailer.html into the PulmoSight trailer MP4 (1920×1080, 30 fps, with soundtrack).

The HTML animation is driven by a single render(t) clock, so every frame is captured deterministically through
Microsoft Edge (Playwright) and encoded with ffmpeg (bundled by imageio-ffmpeg).

    pip install playwright imageio-ffmpeg
    python scripts/render_trailer.py              # → assets/trailer/PulmoSight_trailer.mp4
    python scripts/render_trailer.py --stills     # one PNG per scene, for checking the design quickly
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
TRAILER = ROOT / "assets" / "trailer"
FPS, DURATION = 30, 46
STILL_TIMES = [4.5, 9.5, 13.5, 22.3, 28.8, 35.2, 38.4, 45.0]


def capture(html, out_dir, times, name):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.add_init_script("window.__FRAME_MODE__ = true;")          # disables the live preview loop
        page.goto(html.as_uri()); page.wait_for_load_state("networkidle"); page.wait_for_timeout(500)
        for i, t in enumerate(times):
            page.evaluate(f"render({t})")
            page.screenshot(path=str(out_dir / name(i, t)))
            if i % 150 == 0:
                print(f"  frame {i + 1}/{len(times)}", flush=True)
        browser.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", type=Path, default=TRAILER / "trailer.html")
    ap.add_argument("--out", type=Path, default=TRAILER / "PulmoSight_trailer.mp4")
    ap.add_argument("--stills", action="store_true", help="render one still per scene into assets/trailer/stills/")
    args = ap.parse_args()

    if args.stills:
        out = TRAILER / "stills"; out.mkdir(exist_ok=True)
        capture(args.html, out, STILL_TIMES, lambda i, t: f"still_{t:05.1f}.png")
        print("stills written to", out); return

    audio = TRAILER / "soundtrack.wav"
    if not audio.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_soundtrack.py"), str(audio)], check=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        frames = Path(tmp)
        capture(args.html, frames, [i / FPS for i in range(FPS * DURATION)], lambda i, t: f"f_{i:05d}.png")
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "f_%05d.png"), "-i", str(audio),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(args.out)], check=True)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
