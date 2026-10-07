"""Build the cover-art watermark badge from the minimalist logo (scribe#14).

    uv run python scripts/make_badge.py [assets/scribe-logo-minimalist.jpg] [out.png]

Rebuilt rather than cropped: the source is a JPEG with a textured background and a
wordmark, and a crop would carry both down to a ~70 px corner as smudges. This keeps only
the teal book-and-quill emblem (selected by colour), recolours it to one flat teal, and
draws a fresh navy disk with a white ring around it. The wordmark is dropped because it is
unreadable at corner size. Output is a 512 px RGBA PNG; cover.py scales it down.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

TEAL = (74, 184, 196)   # sampled from the logo's emblem (median of teal pixels)
NAVY = (31, 46, 61)     # sampled from the logo's disk
SIZE = 512
SUPERSAMPLE = 4         # draw at 4x, downsample once: anti-aliased ring and edges
RING = 0.03             # ring width as a fraction of the badge diameter
EMBLEM = 0.62           # emblem's longest side as a fraction of the diameter


def emblem_alpha(src: Image.Image) -> Image.Image:
    """Alpha mask of the teal emblem: how far each pixel's green exceeds its red.

    The navy background sits near g - r = 15 and the teal near g - r = 110, so a linear
    ramp between those maps background to 0 and emblem to 255 with soft JPEG edges kept.
    White (the ring and wordmark) has g - r = 0 and drops out on its own.
    """
    r, g, _ = src.convert("RGB").split()
    diff = ImageChops.subtract(g, r)
    lo, hi = 20, 105
    return diff.point(lambda v: 0 if v <= lo else 255 if v >= hi else (v - lo) * 255 // (hi - lo))


def build(src_path: Path) -> Image.Image:
    alpha = emblem_alpha(Image.open(src_path))
    bbox = alpha.point(lambda v: 255 if v > 128 else 0).getbbox()
    if not bbox:
        raise SystemExit(f"no teal emblem found in {src_path}")
    emblem = alpha.crop(bbox)

    n = SIZE * SUPERSAMPLE
    badge = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge)
    draw.ellipse((0, 0, n - 1, n - 1), fill=NAVY + (255,))
    w = int(n * RING)
    draw.ellipse((w // 2, w // 2, n - 1 - w // 2, n - 1 - w // 2),
                 outline=(255, 255, 255, 255), width=w)

    scale = n * EMBLEM / max(emblem.size)
    emblem = emblem.resize((round(emblem.width * scale), round(emblem.height * scale)),
                           Image.LANCZOS)
    layer = Image.new("RGBA", emblem.size, TEAL + (0,))
    layer.putalpha(emblem)
    badge.alpha_composite(layer, ((n - emblem.width) // 2, (n - emblem.height) // 2))
    return badge.resize((SIZE, SIZE), Image.LANCZOS)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "assets/scribe-logo-minimalist.jpg"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "src/scribe/assets/scribe-badge.png"
    build(src).save(out, optimize=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
