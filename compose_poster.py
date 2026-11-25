#!/usr/bin/env python3
"""
Composite a Blender Cycles transparent-floor render onto a clean studio white
canvas with subtle fine film grain and the classic serif italic title.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
    "/System/Library/Fonts/Supplemental/Baskerville.ttc",
    "/System/Library/Fonts/NewYorkItalic.ttf",
    "/System/Library/Fonts/Times.ttc",
]


def load_serif_italic(size: int):
    for p in FONT_CANDIDATES:
        path = Path(p)
        if path.exists():
            try:
                if path.suffix == ".ttc":
                    # Try italic face index if TTC
                    for idx in (1, 2, 0):
                        try:
                            f = ImageFont.truetype(str(path), size=size, index=idx)
                            if "italic" in " ".join(f.getname()).lower():
                                return f
                        except Exception:
                            pass
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def compose(raw_png: Path, out_png: Path, title: str, grain_strength: float = 2.6):
    fg = Image.open(raw_png).convert("RGBA")
    w, h = fg.size

    # Slight vertical offset upward so there's breathing room for the bottom title
    bg = Image.new("RGBA", (w, h), (250, 250, 250, 255))
    # Paste fg shifted slightly up by ~3% of height
    y_shift = -int(h * 0.03)
    bg.alpha_composite(fg, dest=(0, y_shift))
    img = bg.convert("RGB")

    # Add subtle monochromatic film grain like the reference print
    if grain_strength > 0:
        rng = np.random.default_rng(42)
        arr = np.asarray(img, dtype=np.float32)
        noise = rng.normal(0.0, grain_strength, size=(h, w, 1)).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")

    # Draw serif italic title centered near bottom
    draw = ImageDraw.Draw(img)
    font_size = int(h * 0.038)
    font = load_serif_italic(font_size)
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (w - tw) // 2
    ty = int(h * 0.915) - th // 2
    draw.text((tx, ty), title, fill=(45, 45, 45), font=font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, quality=95)
    print(f"Saved final poster: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_png", type=Path)
    parser.add_argument("out_png", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--grain", type=float, default=2.6)
    args = parser.parse_args()
    compose(args.raw_png, args.out_png, args.title, args.grain)


if __name__ == "__main__":
    main()
