"""Experimental image proxies, NOT sun-occlusion or road-legibility labels.

No GPS, time, projected sun position, camera-wide statistics or labels are used.
Relative peak features follow the local sun_visible_probe draft; flat-image
handling, input checks and relative road features are explicit in this version.
The deployed glare/features.py and its fingerprint remain unchanged.
"""
import cv2
import numpy as np

VERSION = "visibility-proxies-v1"
SUN = ["peak_to_median", "peak_area", "peak_roundness", "peak_ring_ratio", "peak_glow", "peak_chroma"]
ROAD = ["road_contrast_relative", "road_gradient_relative", "road_laplacian_relative", "road_clipped_fraction"]
DIAGNOSTICS = ["peak_x", "peak_y"]


def extract_array(image):
    """Full-frame BGR uint8 -> 640x368; fixed regions, no fitted preprocessing.

    The brightest upper-65% blob can be sky, a lamp or a white building. The
    lower-center ROI can include a dashboard or vehicle. Ratios reduce some
    multiplicative brightness effects, but do not guarantee camera invariance.
    """
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1):
        raise ValueError("Expected a non-empty uint8 BGR image")
    image = cv2.resize(image, (640, 368), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = gray.shape
    top_h = int(h * .65)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0).astype(float)
    top = blurred[:top_h]
    py, px = np.unravel_index(top.argmax(), top.shape)
    peak, median = max(float(top.max()), 1.), max(float(np.median(gray)), 1.)
    yy, xx = np.ogrid[:h, :w]
    distance = np.hypot(xx - px, yy - py)
    disc = distance < .03 * w
    ring = (distance >= .10 * w) & (distance < .15 * w)
    far = (distance >= .10 * w) & (distance < .20 * w)
    area, roundness = 0., 0.
    # A uniform bright region is not evidence of a localized sun/blob.
    if float(top.max() - top.min()) >= 1:
        mask = (top >= .9 * peak).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        component = int(labels[py, px])
        if component:
            area = float(stats[component, cv2.CC_STAT_AREA]) / (h * w)
            contours, _ = cv2.findContours((labels == component).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                roundness = min(4 * np.pi * cv2.contourArea(contour) / perimeter**2, 1.)
    road = gray[int(h*.6):int(h*.85), int(w*.2):int(w*.8)].astype(float)
    p10, p90 = np.percentile(road, [10, 90])
    mean = max(float(road.mean()), 1.)
    gx = cv2.Sobel(road, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(road, cv2.CV_64F, 0, 1, ksize=3)
    return dict(peak_to_median=peak/median, peak_area=area, peak_roundness=roundness,
                peak_ring_ratio=float(gray[ring].mean())/peak,
                peak_glow=(float(gray[disc].mean())-float(gray[far].mean()))/median,
                peak_chroma=float(hsv[..., 1][disc].mean())/255.,
                road_contrast_relative=float((p90-p10)/(p90+p10+1)),
                road_gradient_relative=float(np.hypot(gx, gy).mean())/(8*mean),
                road_laplacian_relative=float(cv2.Laplacian(road, cv2.CV_64F).var())/(mean**2+1),
                road_clipped_fraction=float((road >= 250).mean()), peak_x=float(px/w), peak_y=float(py/h))
