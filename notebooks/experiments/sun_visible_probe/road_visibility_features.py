"""Fixed image proxies for lost road detail; NOT human visibility or luminance.

Literature motivation: veiling contrast loss, foreground underexposure and
road-marking visibility. Pixel thresholds are engineering features, never
clinical/physical grade cutoffs. No labels, metadata, fitted camera statistics,
neural networks or image enhancement are used. Deployed preprocessing is intact.
"""
from functools import lru_cache

import cv2
import numpy as np

VERSION = "road-information-loss-v1"
SIZE = (640, 368)
POLYGON = ((.44, .50), (.56, .50), (.92, .92), (.08, .92))
FEATURES = ["rv_local_std9", "rv_fine_coarse", "rv_contrast_p10", "rv_contrast_p50",
            "rv_edge_density", "rv_long_line", "rv_line_support", "rv_bilateral",
            "rv_white_clip", "rv_black_clip", "rv_halo_overlap", "rv_entropy"]


@lru_cache(maxsize=1)
def road_mask():
    w, h = SIZE
    points = np.rint(np.asarray(POLYGON) * [w-1, h-1]).astype(np.int32)
    result = np.zeros((h, w), np.uint8)
    cv2.fillPoly(result, [points], 1)
    result.flags.writeable = False
    return result


def analyze(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) < 1):
        raise ValueError("Expected nonempty uint8 BGR image")
    image = cv2.resize(image, SIZE, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = road_mask().astype(bool)
    g = gray.astype(np.float32)
    w, h = SIZE
    def local_std(size):
        average = cv2.blur(g, (size, size))
        return np.sqrt(np.maximum(cv2.blur(g*g, (size, size)) - average*average, 0))
    fine, coarse = local_std(9), local_std(41)
    # Block percentiles describe the distribution of local contrast, rather
    # than one global standard deviation that conflates texture and shading.
    def tiles(a):
        return a[:h//16*16, :w//16*16].reshape(h//16, 16, w//16, 16).transpose(0, 2, 1, 3).reshape(-1, 256)
    blocks = tiles(gray)[tiles(mask).mean(axis=1) >= .75]
    lo, hi = np.percentile(blocks, [10, 90], axis=1)
    contrast = (hi-lo)/(hi+lo+1)
    edges = cv2.Canny(gray, 50, 150)
    road_edges = np.where(mask, edges, 0).astype(np.uint8)
    lines = cv2.HoughLinesP(road_edges, 1, np.pi/180, threshold=20, minLineLength=32, maxLineGap=16)
    line_mask = np.zeros_like(gray)
    longest, left, right = 0., False, False
    for x1, y1, x2, y2 in [] if lines is None else np.asarray(lines).reshape(-1, 4):
        dx, dy = int(x2-x1), int(y2-y1)
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        if not 25 <= angle <= 85 or not dy:
            continue
        top_x = x1 + (.5*h-y1)*dx/dy
        bottom_x = x1 + (.9*h-y1)*dx/dy
        if not .2*w <= top_x <= .8*w or not 0 <= bottom_x <= w:
            continue
        left |= bottom_x < w/2
        right |= bottom_x >= w/2
        longest = max(longest, float(np.hypot(dx, dy))/np.hypot(w, h))
        cv2.line(line_mask, (x1, y1), (x2, y2), 1, 2)
    edge_count = int(np.count_nonzero(road_edges))
    hist = np.histogram(gray[mask], bins=16, range=(0, 256))[0].astype(float)
    prob = hist[hist > 0] / hist.sum()
    halo = cv2.GaussianBlur(gray, (41, 41), 0) >= 230
    f, c = float(fine[mask].mean()), float(coarse[mask].mean())
    features = dict(zip(FEATURES, [f/255, f/(f+c+1), float(np.quantile(contrast, .1)), float(np.median(contrast)),
                    float((road_edges[mask] > 0).mean()), longest,
                    float(np.count_nonzero((line_mask > 0) & (road_edges > 0)))/max(edge_count, 1),
                    float(left and right), float((gray[mask] >= 250).mean()),
                    float((gray[mask] <= 8).mean()), float(halo[mask].mean()), float(-(prob*np.log2(prob)).sum()/4)]))
    if not np.isfinite(list(features.values())).all():
        raise AssertionError("Nonfinite road proxy")
    return features, dict(image=image, mask=mask, edges=road_edges, lines=line_mask)


def extract_array(image):
    return analyze(image)[0]
