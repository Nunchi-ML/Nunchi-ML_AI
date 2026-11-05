"""Image-region shape/detail proxies, not semantic object visibility.

Regions can contain sky, trees or no objects. A zero feature does not mean an
object is hidden. No labels, fitted statistics, neural features or physical
luminance interpretation. Deployed preprocessing remains untouched.
"""
from functools import lru_cache

import cv2
import numpy as np

VERSION = "forward-region-detail-v1"
SIZE = (640, 368)
REGIONS = {"front": [(.20, .28, .80, .68)],
           "context": [(.02, .28, .18, .68), (.82, .28, .98, .68)]}
MEASURES = ["std_p20", "std_p50", "weak_edges", "edge_persistence", "contour_support",
            "corner_coverage", "bright_flat", "dark_flat"]
FEATURES = ["fd_"+region+"_"+measure for region in REGIONS for measure in MEASURES]
FEATURES += ["fd_front_context_contrast", "fd_front_context_edges"]


@lru_cache(maxsize=1)
def masks():
    w, h = SIZE
    result = {}
    for name, rectangles in REGIONS.items():
        mask = np.zeros((h, w), np.uint8)
        for x0, y0, x1, y1 in rectangles:
            mask[round(y0*h):round(y1*h), round(x0*w):round(x1*w)] = 255
        mask.flags.writeable = False
        result[name] = mask
    return result


@lru_cache(maxsize=2)
def cell_count(name):
    y, x = np.where(masks()[name] > 0)
    return len(np.unique((y//16)*(SIZE[0]//16)+x//16))


def extract_array(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) < 1):
        raise ValueError("Expected nonempty uint8 BGR image")
    gray = cv2.cvtColor(cv2.resize(image, SIZE, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
    values = gray.astype(np.float32)
    mean = cv2.blur(values, (9, 9))
    std = np.sqrt(np.maximum(cv2.blur(values*values, (9, 9))-mean*mean, 0))
    weak = cv2.Canny(gray, 25, 75) > 0
    strong = cv2.dilate(cv2.Canny(gray, 75, 200), np.ones((3, 3), np.uint8)) > 0
    result = {}
    for name, mask in masks().items():
        area = mask > 0
        edges = weak & area
        count = int(edges.sum())
        contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        supported = np.zeros_like(gray)
        for contour in contours:
            _, _, w, h = cv2.boundingRect(contour)
            if w >= 8 and h >= 8 and cv2.arcLength(contour, False) >= 48:
                cv2.drawContours(supported, [contour], -1, 1, 1)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=256, qualityLevel=.01, minDistance=4, mask=mask)
        occupied = set() if corners is None else {(int(x)//16, int(y)//16) for x, y in corners.reshape(-1, 2)}
        measures = [float(np.quantile(std[area], .2))/127.5,
                    float(np.median(std[area]))/127.5,
                    float(edges[area].mean()), float((edges & strong).sum())/max(count, 1),
                    float((edges & (supported > 0)).sum())/max(count, 1),
                    len(occupied)/max(cell_count(name), 1),
                    float(((gray >= 220) & (std < 8))[area].mean()),
                    float(((gray <= 16) & (std < 8))[area].mean())]
        result.update({"fd_"+name+"_"+measure: float(np.clip(value, 0, 1)) for measure, value in zip(MEASURES, measures)})
    f, c = result["fd_front_std_p50"], result["fd_context_std_p50"]
    result["fd_front_context_contrast"] = f/(f+c+1/255)
    f, c = result["fd_front_weak_edges"], result["fd_context_weak_edges"]
    result["fd_front_context_edges"] = f/(f+c+1e-6)
    if list(result) != FEATURES or not np.isfinite(list(result.values())).all():
        raise AssertionError("Invalid fixed feature order/values")
    return result
