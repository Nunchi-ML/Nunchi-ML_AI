"""Additional pixel statistics, never physical luminance or a sensor reading."""
import cv2
import numpy as np

EXTRA = ["px_mean", "px_std", "px_p90", "px_p95", "px_p99", "px_bright_area",
         "px_source_contrast", "px_bright_x", "px_bright_y", "px_local_contrast",
         *[f"px_hist_{i}" for i in range(8)]]


def extract_array(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) < 1):
        raise ValueError("Expected nonempty uint8 BGR image")
    image = cv2.resize(image, (640, 368), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    q90, q95, q99 = np.percentile(gray, [90, 95, 99])
    bright = gray >= max(q95, 1)
    yy, xx = np.nonzero(bright)
    background = gray[~bright]
    contrast = ((float(gray[bright].mean()) - float(background.mean())) / 255
                if bright.any() and len(background) else 0.)
    mean = cv2.blur(gray, (15, 15))
    local_variance = np.maximum(cv2.blur(gray * gray, (15, 15)) - mean * mean, 0)
    hist = np.histogram(gray, bins=8, range=(0, 256))[0] / gray.size
    values = [float(gray.mean()) / 255, float(gray.std()) / 255, q90 / 255, q95 / 255, q99 / 255,
              float(bright.mean()), contrast, float(xx.mean()) / 640 if len(xx) else .5,
              float(yy.mean()) / 368 if len(yy) else .5, float(np.sqrt(local_variance).mean()) / 255, *hist]
    return dict(zip(EXTRA, map(float, values)))
