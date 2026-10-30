"""전방 화면 한 장에서 눈부심 관련 숫자 12개를 뽑는다 (딥러닝 없음, OpenCV).

사용:
    python -m src.glare.features data/glare data/glare_features.csv

data/glare 아래 GLARE 데이터셋 구조(Images/vidN/<영상>_annotations/*.png)를 그대로 둔다고 가정한다.
"""
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

FEATURES = [
    "white_ratio",      # 하얗게 날아간 픽셀 비율 (밝기 > 245)
    "bright_blob",      # 크게 번진 밝은 덩어리 비율 (흐린 화면에서 > 230)
    "mean_all",         # 화면 전체 평균 밝기
    "sky_mean",         # 위쪽 45% 평균 밝기
    "road_mean",        # 도로 부분(아래 60~85%, 가운데 60%) 평균 밝기
    "road_contrast",    # 도로 부분 밝기 표준편차 = 대비
    "sky_road_ratio",   # 하늘 / 도로 밝기 비 (역광 정도)
    "road_sharp",       # 도로 부분 라플라시안 분산 = 윤곽 선명도
    "low_sat",          # 채도 < 25 비율 = 색이 빠진 희뿌연 정도
    "brightest_x",      # 가장 밝은 점의 가로 위치 (0 왼쪽 ~ 1 오른쪽)
    "brightest_y",      # 가장 밝은 점의 세로 위치 (0 위 ~ 1 아래)
    "brightest_val",    # 가장 밝은 점의 밝기
]


def extract(path):
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    im = cv2.resize(im, (640, 368))
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    sky = g[: int(h * 0.45)]
    road = g[int(h * 0.6): int(h * 0.85), int(w * 0.2): int(w * 0.8)]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(g, (41, 41), 0)
    by, bx = np.unravel_index(blur.argmax(), g.shape)
    return dict(
        white_ratio=(g > 245).mean(),
        bright_blob=(blur > 230).mean(),
        mean_all=g.mean(),
        sky_mean=sky.mean(),
        road_mean=road.mean(),
        road_contrast=road.std(),
        sky_road_ratio=sky.mean() / max(road.mean(), 1),
        road_sharp=cv2.Laplacian(road, cv2.CV_64F).var(),
        low_sat=(hsv[..., 1] < 25).mean(),
        brightest_x=bx / w,
        brightest_y=by / h,
        brightest_val=blur.max(),
    )


def main(root, out):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "Images", "*", "*", "*.png"))):
        parts = p.replace("\\", "/").split("/")
        r = extract(p)
        r.update(vid=parts[-3], file=parts[-1])
        rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(rows)}장 → {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
