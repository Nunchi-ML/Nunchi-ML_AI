"""사진 한 장 안에서만 계산하는 '해가 보이는가' 특징 12개 (카메라 노출 · 밝기 척도에 덜 흔들리게).

기존 화면 숫자 12개(features.py)는 절대 밝기 문턱(> 245, > 230)을 써서 카메라마다 밝기가 다르면 무너졌다.
여기서는 모든 값을 '그 사진의 중앙값 · 최고 밝기' 에 대한 비율로 만든다. 라벨 · 해 위치는 쓰지 않는다.

    peak_med   위쪽 65% 에서 가장 밝은 점(흐림 후) / 사진 중앙값
    dyn        밝기 99 백분위 / 중앙값
    blob_area  최고 밝기의 90% 이상인 덩어리(최고점이 속한)의 면적 비율
    blob_circ  그 덩어리의 둥근 정도 (해 원반은 둥긂, 하늘 · 흰 벽은 아님)
    sat_top    위쪽 65% 에서 밝기 ≥ 250 인 비율 (디스크가 날아간 정도)
    ring1~3    최고점에서 멀어질수록 밝기가 얼마나 남나 (번짐) — 반경 0.04 · 0.10 · 0.20 폭 둘레 평균 / 최고 밝기
    glow_con   최고점 근처 원 평균 − 먼 고리 평균, 중앙값으로 나눔
    disc_sat   최고점 근처 원의 채도 (해 원반은 하양 = 채도 낮음)
    peak_x, peak_y  최고점 위치 (0~1)

    python notebooks/experiments/sun_visible_probe/rel_features.py        # 624장 → data/sun_visible/rel_features.csv
"""
import cv2
import numpy as np

REL = ["peak_med", "dyn", "blob_area", "blob_circ", "sat_top", "ring1", "ring2", "ring3", "glow_con", "disc_sat",
       "peak_x", "peak_y"]


def rel_array(im):
    im = cv2.resize(im, (640, 368), interpolation=cv2.INTER_LINEAR)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, w = g.shape
    top_h = int(h * 0.65)
    med = float(np.median(g)) + 1.0
    blur = cv2.GaussianBlur(g, (21, 21), 0).astype(np.float32)
    top = blur[:top_h]
    py, px = np.unravel_index(top.argmax(), top.shape)
    peak = float(top.max()) + 1e-6

    mask = (top >= 0.9 * peak).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    k = lab[py, px]
    area = stats[k, cv2.CC_STAT_AREA] if k else 1
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    per = max(cv2.arcLength(max(cnts, key=cv2.contourArea), True), 1.0) if cnts else 1.0
    circ = float(min(4 * np.pi * area / per ** 2, 1.0))

    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(xx - px, yy - py)

    def ring(r0, r1):
        m = (dist >= r0 * w) & (dist < r1 * w)
        return float(g[m].mean()) / peak if m.any() else 0.0

    disc = dist < 0.03 * w
    far = (dist >= 0.10 * w) & (dist < 0.20 * w)
    glow = (float(g[disc].mean()) - float(g[far].mean())) / med if far.any() else 0.0
    return dict(peak_med=peak / med, dyn=float(np.percentile(g, 99)) / med,
                blob_area=area / (h * w), blob_circ=circ, sat_top=float((g[:top_h] >= 250).mean()),
                ring1=ring(0.04, 0.06), ring2=ring(0.10, 0.15), ring3=ring(0.20, 0.30),
                glow_con=glow, disc_sat=float(hsv[..., 1][disc].mean()) / 255.0,
                peak_x=px / w, peak_y=py / h)


def rel_file(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"cannot read {path}")
    return rel_array(im)
