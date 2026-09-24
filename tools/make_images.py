#!/usr/bin/env python3
"""
Draws the site's raster images so they match index.html (the lattice sphere):

  apple-touch-icon.png  180x180   solid #0a0b0f, sphere ~70% of the width
  icon-512.png          512x512   same design, for manifests
  og.png                1200x630  sphere with faint neighbour edges, wordmark + formula lower left
  favicon.ico           32x32 + 16x16, transparent, accent-coloured points

The look (not the code) of the page is reproduced: a Fibonacci sphere of warm-white points, orthographic
projection, points nearer the viewer brighter and slightly larger, a soft additive glow around each point,
a faint radial halo behind the sphere, a handful of ember-coloured points, and (big image only) short faint
edges between lattice neighbours.

Deterministic: fixed seed, no wall-clock input.  Run from anywhere:

    python3 tools/make_images.py

Needs Pillow (tested with 9.3) and numpy.  Output goes next to index.html (the parent of this folder).
"""
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)

SEED = 20260922                    # same seed the page uses for its object
BG = (10, 11, 15)                  # #0a0b0f
WHITE = (255, 240, 219)            # warm white at full brightness
ACCENT = (235, 162, 90)            # #eba25a
FG = (217, 214, 207)               # #d9d6cf
FG_MID = (179, 176, 169)           # #b3b0a9

SANS_FONTS = [
    ("/System/Library/Fonts/SFNS.ttf", 0),
    ("/System/Library/Fonts/Helvetica.ttc", 0),
    ("/Library/Fonts/Arial.ttf", 0),
]
SERIF_ITALIC_FONTS = [
    ("/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf", 0),
    ("/System/Library/Fonts/Times.ttc", 2),          # index 2 is the italic face (1 is bold)
]
SERIF_UPRIGHT_FONTS = [
    ("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 0),
    ("/System/Library/Fonts/Times.ttc", 0),
]

GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))
FIB_OFFSETS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]


# ----------------------------------------------------------------------------- geometry

def fibonacci_sphere(n):
    """n points spread evenly over the unit sphere (same construction as the page)."""
    pts = []
    for i in range(n):
        y = 1 - (2 * (i + 0.5)) / n
        r = math.sqrt(max(0.0, 1 - y * y))
        th = GOLDEN_ANGLE * i
        pts.append((math.cos(th) * r, y, math.sin(th) * r))
    return pts


def build_edges(pts, k=3):
    """Short edges to the k nearest lattice neighbours; candidates sit at Fibonacci index offsets."""
    n = len(pts)
    max_d2 = 2.6 * 14.51 / n
    edges = set()
    for i, (x, y, z) in enumerate(pts):
        cand = []
        for f in FIB_OFFSETS:
            if f >= n:
                break
            for j in (i + f, i - f):
                if 0 <= j < n:
                    px, py, pz = pts[j]
                    d2 = (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2
                    if d2 <= max_d2:
                        cand.append((d2, j))
        cand.sort()
        for _, j in cand[:k]:
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def rotate(p, ry, rx):
    """Rotate about Y, then about X (tilts the pole spiral away from the viewer)."""
    x, y, z = p
    cy, sy = math.cos(ry), math.sin(ry)
    x1, z1 = x * cy + z * sy, -x * sy + z * cy
    cx, sx = math.cos(rx), math.sin(rx)
    y2, z2 = y * cx - z1 * sx, y * sx + z1 * cx
    return (x1, y2, z2)


def smoothstep(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def depth_fade(z):
    """Brightness by depth, as in the page's point shader (z = +1 nearest the viewer)."""
    return 0.22 + 0.78 * smoothstep(-1.3, 1.3, z)


def mix(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


# ----------------------------------------------------------------------------- rendering

def render_sphere(size, center, radius, n, dot_d, ss=3, seed=SEED, edge_opacity=0.0,
                  halo=0.08, glow=0.55, hot_frac=0.04, rot=(0.9, 0.26)):
    """
    RGB image of `size` with the sphere drawn on BG.
      center, radius  in final pixels        n        number of lattice points
      dot_d           core dot diameter, px  ss       supersampling factor
      edge_opacity    0 for none             halo     strength of the radial glow behind the sphere
      glow            per-point glow gain    hot_frac share of accent-coloured points
    """
    w, h = size
    W, H = w * ss, h * ss
    cx, cy = center[0] * ss, center[1] * ss
    R = radius * ss
    rnd = random.Random(seed)

    base = fibonacci_sphere(n)
    pts = [rotate(p, *rot) for p in base]
    proj = [(cx + x * R, cy - y * R, z) for (x, y, z) in pts]   # z: +1 nearest the viewer
    attrs = []
    for _ in range(n):
        s = 0.7 + 0.6 * rnd.random()
        hot = rnd.random() < hot_frac
        phase = rnd.random() * 2 * math.pi
        attrs.append((s, hot, phase))

    acc = np.empty((H, W, 3), dtype=np.float32)
    acc[...] = np.array(BG, dtype=np.float32)

    # radial halo behind the object: pow(1 - r, 2.6) out to 2.6 R, warm white with a hint of ember
    if halo > 0:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (2.6 * R)
        a = np.clip(1.0 - r, 0.0, 1.0) ** 2.6 * halo
        acc += a[..., None] * np.array(mix(WHITE, ACCENT, 0.12), dtype=np.float32)
        del xx, yy, r, a

    # faint edges between lattice neighbours, back to front, dimmer at the back
    if edge_opacity > 0:
        lay = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(lay)
        segs = sorted(((proj[a][2] + proj[b][2]) * 0.5, a, b) for a, b in build_edges(base, 3))
        lw = max(1, int(round(0.5 * ss)))
        for zm, a, b in segs:
            f = 0.15 + 0.85 * smoothstep(-1.3, 1.3, zm)
            d.line([proj[a][:2], proj[b][:2]], fill=int(round(255 * edge_opacity * f)), width=lw)
        acc += (np.asarray(lay, dtype=np.float32) / 255.0)[..., None] * np.array(WHITE, dtype=np.float32)

    # points: a crisp core plus a wide soft glow (the glow is a blurred, larger copy of the cores)
    core = Image.new("RGB", (W, H), (0, 0, 0))
    glow_im = Image.new("RGB", (W, H), (0, 0, 0))
    dc, dg = ImageDraw.Draw(core), ImageDraw.Draw(glow_im)
    for i in sorted(range(n), key=lambda i: proj[i][2]):          # back to front
        x, y, z = proj[i]
        s, hot, phase = attrs[i]
        fade = depth_fade(z)
        tw = 0.86 + 0.14 * math.sin(1.7 * 3.0 + phase)             # the page's twinkle, frozen
        persp = 0.88 + 0.24 * (z + 1) * 0.5                         # nearer points slightly larger
        r = 0.5 * dot_d * ss * s * persp * (1.6 if hot else 1.0)
        col = ACCENT if hot else WHITE
        a = fade * tw * 0.9
        dc.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(round(c * a * 0.5)) for c in col))
        ri = r * 0.62
        dc.ellipse([x - ri, y - ri, x + ri, y + ri], fill=tuple(int(round(c * a)) for c in col))
        rg = r * 2.6
        g = fade * fade                                             # glow dies off faster at the back
        dg.ellipse([x - rg, y - rg, x + rg, y + rg], fill=tuple(int(round(c * g)) for c in col))
    glow_im = glow_im.filter(ImageFilter.GaussianBlur(1.4 * dot_d * ss))
    acc += np.asarray(glow_im, dtype=np.float32) * glow
    acc += np.asarray(core, dtype=np.float32)

    out = Image.fromarray(np.clip(acc + 0.5, 0, 255).astype(np.uint8), "RGB")
    if ss > 1:
        out = out.resize((w, h), Image.Resampling.LANCZOS)
    return out


def load_font(candidates, size, fallback=None):
    for path, idx in candidates:
        try:
            return ImageFont.truetype(path, size, index=idx)
        except OSError:
            continue
    if fallback is not None:
        return fallback
    return ImageFont.load_default()


def draw_text(im, wordmark_px=30, formula_px=26, left=72, bottom=72):
    """Wordmark in the accent colour, the formula under it set like the page: variables italic,
    the operator and "log" upright and a little dimmer."""
    d = ImageDraw.Draw(im)
    sans = load_font(SANS_FONTS, wordmark_px)
    try:                                                   # SF is a variable font: pin the regular weight
        sans.set_variation_by_axes([100, 28, 400, 400])
    except Exception:
        pass
    it = load_font(SERIF_ITALIC_FONTS, formula_px, fallback=sans)
    up = load_font(SERIF_UPRIGHT_FONTS, formula_px, fallback=it)
    up_small = load_font(SERIF_UPRIGHT_FONTS, int(round(formula_px * 0.82)), fallback=up)

    h = im.size[1]
    y_formula = h - bottom                                # baseline of the formula
    y_word = y_formula - int(round(formula_px * 0.66)) - int(round(wordmark_px * 0.8))
    d.text((left, y_word), "Entropy Partners", font=sans, fill=ACCENT, anchor="ls")

    em = formula_px
    x = left
    pieces = [
        ("S", it, FG, 0.0),
        ("=", up, FG_MID, 0.22 * em),
        ("k", it, FG, 0.22 * em),
        ("log", up_small, FG_MID, 0.14 * em),
        ("W", it, FG, 0.12 * em),
    ]
    for text, font, colour, gap in pieces:
        x += gap
        d.text((x, y_formula), text, font=font, fill=colour, anchor="ls")
        x += font.getlength(text)
    return im


def save_png(im, path, limit=None):
    im.save(path, "PNG", optimize=True)
    if limit is not None and os.path.getsize(path) > limit:
        # palette version: 256 colours with error diffusion keeps the glow smooth
        q = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
        q.save(path, "PNG", optimize=True)


# ----------------------------------------------------------------------------- the images

def make_icon(px):
    """Square icon: solid ground, sphere ~70% of the width, no edges (they vanish at icon sizes)."""
    return render_sphere((px, px), (px / 2, px / 2), 0.35 * px, n=420, dot_d=px * 2.8 / 180,
                         ss=4, halo=0.06, glow=0.2)


def make_favicon_frame(px):
    """ICO frame: fewer, larger points so it still reads as a speckled globe at 16-32 px.

    Transparent ground, every point in the accent colour: a tab strip may be light or dark, and amber
    reads on both where warm white would vanish on a light one. The sphere is rendered on black and its
    brightness becomes the alpha channel."""
    n = 120 if px >= 32 else 90
    lit = render_sphere((px, px), (px / 2, px / 2), 0.47 * px, n=n, dot_d=max(1.0, px * 0.055),
                        ss=8, halo=0.0, glow=0.2)
    lum = np.asarray(lit, dtype=np.float32).max(axis=2) / 255.0
    floor = max(BG) / 255.0                       # the ground itself is not light: fully transparent
    alpha = np.clip((lum - floor) / (1.0 - floor) * 1.25, 0.0, 1.0)
    rgba = np.zeros(lum.shape + (4,), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = ACCENT
    rgba[..., 3] = (alpha * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def make_og():
    w, h = 1200, 630
    im = render_sphere((w, h), (700, 315), 220, n=1500, dot_d=3.0, ss=2,
                       edge_opacity=0.10, halo=0.06, glow=0.2)
    return draw_text(im)


def main():
    out = SITE
    jobs = []

    im = make_icon(180)
    p = os.path.join(out, "apple-touch-icon.png")
    save_png(im, p)
    jobs.append(p)

    im = make_icon(512)
    p = os.path.join(out, "icon-512.png")
    save_png(im, p)
    jobs.append(p)

    im = make_og()
    p = os.path.join(out, "og.png")
    save_png(im, p, limit=400_000)
    jobs.append(p)

    f32, f16 = make_favicon_frame(32), make_favicon_frame(16)
    p = os.path.join(out, "favicon.ico")
    f32.save(p, format="ICO", sizes=[(32, 32), (16, 16)], append_images=[f16])
    jobs.append(p)

    for p in jobs:
        with Image.open(p) as chk:
            extra = ""
            if p.endswith(".ico"):
                extra = "  frames " + ", ".join("%dx%d" % s for s in sorted(chk.ico.sizes()))
            print("%-22s %s %s  %d bytes%s" % (os.path.basename(p), chk.size, chk.mode, os.path.getsize(p), extra))


if __name__ == "__main__":
    main()
