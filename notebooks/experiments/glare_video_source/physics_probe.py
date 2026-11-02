"""물리 특징 실험: 눈부심 각도 θ · 눈부심 세기 지수 · 시정 · 강수.

    θ          시선(진행 방향 · 수평)과 해 사이 실제 각도 = arccos(cos 해높이 · cos Δ)
    glare_idx  일사량 / θ²  — CIE 불능 눈부심 식(베일 휘도 ∝ 눈부심 조도 / θ²) 을 본뜬 지수.
               해가 시선에 가깝고(θ 작음) 햇빛이 셀수록 큼. θ 는 1° 아래로 안 내림
    visibility 시정 (안개 · 미세먼지로 빛이 퍼짐) · rain 강수 (젖은 도로 반사)
방향 기록이 틀린 사진(heading_suspect)은 θ · glare_idx 빈칸.

비교 (한국 284장, 촬영 단위 GroupKFold(5) — asos_probe.py 와 같은 사진 · 같은 분할):
    해 위치 + 날씨 (지금 최고) · θ + 지수 (2개만) · 물리 전부 · 화면 12 + 물리 전부

    python notebooks/experiments/glare_video_source/physics_probe.py
"""
import numpy as np
import pandas as pd

from nunchi_ml.glare.features import FEATURES

import asos_probe as AP
import fetch_asos as A
import find_sun_frames as F
import v2_candidate as V

OUT = F.OUT
SUN, WX = V.SUN, AP.WX
PHYS = ["theta", "glare_idx"]
WX2 = ["visibility", "rain", "radiation"]


def physics(d):
    alt, gap = np.radians(d.sun_alt), np.radians(d.gap)
    theta = np.degrees(np.arccos((np.cos(alt) * np.cos(gap)).clip(-1, 1)))
    theta = theta.where(~d.heading_suspect.astype(bool) & (d.sun_alt > 0))
    return d.assign(theta=theta, glare_idx=d.radiation.fillna(0) / theta.clip(lower=1) ** 2)


def main():
    k = V.korea_with_sun()
    t = pd.concat([pd.read_csv(OUT / n)[["id", "captured_at_kst"]] for n in ("labels_drive.csv", "labels_dash.csv")]).drop_duplicates("id")
    k = physics(A.attach(k.merge(t, on="id")))

    print("θ 구간별 눈부심 (맑은 날 · 방향 맞는 사진)")
    c = k[(k.cloud <= 2) & k.theta.notna()]
    print(c.groupby(pd.cut(c.theta, [0, 10, 20, 30, 45, 90, 180])).label.agg(
        사진="size", 눈부심=lambda x: (x >= 1).mean(), 심함=lambda x: (x == 2).mean()).round(2).to_string())
    print("\n눈부심 여부와 순위 상관:", {f: round(k[[f]].assign(g=k.label >= 1).corr("spearman").iloc[0, 1], 2)
                                  for f in PHYS + WX + WX2})

    print("\n모델 비교 (한국만, 촬영 단위 GroupKFold 5)")
    cams = ("블랙박스", "도로 광각")
    for name, cols in (("해 위치 + 날씨 (지금 최고)", SUN + WX),
                       ("θ + 눈부심 지수 (2개만)", PHYS),
                       ("θ + 지수 + 날씨", PHYS + WX),
                       ("물리 전부 (해 위치 · θ · 지수 · 날씨 · 시정 · 강수)", SUN + PHYS + WX + WX2),
                       ("화면 12 + 물리 전부", FEATURES + SUN + PHYS + WX + WX2)):
        AP.show(name, k, AP.cv(k, cols), cams)


if __name__ == "__main__":
    main()
