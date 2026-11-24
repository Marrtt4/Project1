"""
CPU quality gate for generated images (numpy + Pillow only, no GPU).

check() looks at one image and returns (ok, reasons, metrics, signature).
If ok is False the app throws the image away and renders a new one.

Tests run on a downscaled copy (max 256 px) so they are fast and consistent.
The numbers below are heuristics. If good images get rejected too often, lower
the thresholds in LEVELS; if bad ones slip through, raise them.
"""
import numpy as np
from PIL import Image

LEVELS = {
    "relaxed": dict(min_sharp=2.0, max_sharp=20000.0, min_contrast=0.025, min_mean=0.03,
                    max_mean=0.97, max_clip=0.85, min_entropy=1.2, min_diff=0.006,
                    max_palette=None, min_corr=0.35),
    "normal": dict(min_sharp=4.0, max_sharp=20000.0, min_contrast=0.040, min_mean=0.05,
                   max_mean=0.95, max_clip=0.75, min_entropy=1.8, min_diff=0.012,
                   max_palette=None, min_corr=0.50),
    "strict": dict(min_sharp=8.0, max_sharp=20000.0, min_contrast=0.060, min_mean=0.08,
                   max_mean=0.92, max_clip=0.65, min_entropy=2.4, min_diff=0.025,
                   max_palette=0.38, min_corr=0.60),
}


def _hex_to_rgb(value):
    value = value.lstrip("#")
    return [int(value[i:i + 2], 16) for i in (0, 2, 4)]


def analyze(image, palette=None):
    """Measure an image. Returns (metrics dict, 16x16 signature array)."""
    rgb = image.convert("RGB")
    rgb.thumbnail((256, 256), Image.LANCZOS)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = np.nan_to_num(arr)
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

    # Sharpness = variance of the Laplacian (low = blurry / smeared, huge = noise).
    g = lum * 255.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    sharp = float(lap.var())

    # Neighbour-pixel correlation: real images (even very detailed ones) score ~0.85+,
    # pure static scores ~0.1. Sharpness cannot tell them apart.
    def _lag_corr(a, b):
        a = a.ravel() - lum.mean()
        b = b.ravel() - lum.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / den) if den > 0 else 1.0
    corr = min(_lag_corr(lum[:, :-1], lum[:, 1:]), _lag_corr(lum[:-1], lum[1:]))

    hist, _ = np.histogram(lum, bins=32, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    metrics = {
        "sharpness": round(sharp, 1),
        "contrast": round(float(lum.std()), 3),
        "brightness": round(float(lum.mean()), 3),
        "clipped": round(float(max((lum < 0.03).mean(), (lum > 0.97).mean())), 3),
        "entropy": round(entropy, 2),
        "neighbor_corr": round(corr, 3),
        "palette_distance": None,
    }

    if palette:
        pal = np.array([_hex_to_rgb(c) for c in palette], dtype=np.float32) / 255.0
        small = np.asarray(rgb.resize((64, 64)), dtype=np.float32).reshape(-1, 3) / 255.0
        dist = np.sqrt(((small[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        metrics["palette_distance"] = round(float(dist.mean() / np.sqrt(3)), 3)

    sig = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).resize((16, 16)),
                     dtype=np.float32) / 255.0
    return metrics, sig


def check(image, level="normal", palette=None, accepted_sigs=None):
    """Return (ok, reasons, metrics, signature)."""
    t = LEVELS.get(level, LEVELS["normal"])
    m, sig = analyze(image, palette)
    reasons = []

    # Low entropy alone is normal for a product on a plain backdrop, so it only
    # counts as "flat" when the contrast is low too.
    if m["contrast"] < t["min_contrast"] or (m["entropy"] < t["min_entropy"] and m["contrast"] < 0.10):
        reasons.append(f"blank or flat image (contrast {m['contrast']}, entropy {m['entropy']})")
    if m["brightness"] < t["min_mean"]:
        reasons.append(f"too dark (brightness {m['brightness']})")
    if m["brightness"] > t["max_mean"]:
        reasons.append(f"washed out (brightness {m['brightness']})")
    if m["clipped"] > t["max_clip"]:
        reasons.append(f"mostly pure black or white ({int(m['clipped'] * 100)}% clipped)")
    if m["sharpness"] < t["min_sharp"]:
        reasons.append(f"too blurry (sharpness {m['sharpness']})")
    if m["neighbor_corr"] < t["min_corr"] or m["sharpness"] > t["max_sharp"]:
        reasons.append(f"noisy or garbled (static-like, pixel correlation {m['neighbor_corr']})")
    if t["max_palette"] is not None and m["palette_distance"] is not None \
            and m["palette_distance"] > t["max_palette"]:
        reasons.append(f"colors drift from the palette (distance {m['palette_distance']})")

    for other in accepted_sigs or []:
        if float(np.abs(sig - other).mean()) < t["min_diff"]:
            reasons.append("too similar to an image already accepted")
            break

    return (not reasons), reasons, m, sig
