"""해 위치-사진 일치도 — G 가 계산한 해의 화면 위치(u · v)와 사진에서 실제로 가장 밝은 점(peak_x · peak_y)이 가까운가.

해가 예상 자리 근처에 실제로 있으면 진짜 눈부심, 엉뚱한 곳이거나 없으면 건물 가림 · 화면 밖 빛으로 본다는 가설.
지평선은 '화면 가운데 높이' 로 가정 (g_sunpatch_probe 와 같은 가정, 폰 기울기를 모르므로 오차 있음).
    예상 위치  x = 0.5 + u/2,  y = 0.5 − v/2   (u · v 는 화면 밖이면 ±1.5 · −0.5~1.5 로 자름)
    sun_dist   예상 위치와 가장 밝은 점의 화면 거리 (가로 · 세로 정규화)
    sun_match  in_fov × exp(−sun_dist / 0.2)

Codex 프로토콜(codex_protocol_compare) 위에 얹는다. 라벨 · 시험 카메라 통계를 쓰지 않는다.

    python notebooks/experiments/sun_visible_probe/match_probe.py --snapshot ... --rel ... --cache ... --out ...
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import codex_protocol_compare as P
import g_visibility_probe as C
from g_weather_time_probe import G_COLUMNS, replay
from nunchi_ml.glare import visibility as V
from rel_features import REL

MATCH = ["sun_dist", "sun_match"]


def add_match(d):
    u = d.u.clip(-1.5, 1.5)
    v = d.v.clip(-0.5, 1.5)
    x, y = 0.5 + u / 2, 0.5 - v / 2
    d["sun_dist"] = np.hypot(d.peak_x - x, d.peak_y - y)
    d["sun_match"] = d.in_fov * np.exp(-d.sun_dist / 0.2)
    return d


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--snapshot", type=Path, required=True)
    a.add_argument("--rel", type=Path, required=True)
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--data-root", type=Path, default=Path("data"))
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--repeats", type=int, default=30)
    a = a.parse_args()
    snap = replay(pd.read_csv(a.snapshot, dtype={"id": str}), a.data_root / "asos", "previous")
    data = C.features(snap, C.image_paths(snap, a.data_root), a.cache)
    rel = pd.read_csv(a.rel, dtype={"id": str})[["id"] + REL].rename(columns={c: "r_" + c for c in REL})
    data = add_match(data.merge(rel, on="id", how="left", validate="one_to_one"))

    print("일치도 분포 (라벨별 중앙값 · 카메라별) — 정상(0)은 멀고 눈부심(1+)은 가까워야 가설이 맞다")
    for cam, g in data.groupby("camera"):
        m0, m1 = g[g.label == 0], g[g.label >= 1]
        print(f"  {cam:8s} sun_dist 정상 {m0.sun_dist.median():.2f} ({len(m0)}) · 눈부심 {m1.sun_dist.median():.2f} ({len(m1)})"
              f" | in_fov 정상 {m0.in_fov.mean():.2f} · 눈부심 {m1.in_fov.mean():.2f}")

    P.METHODS = {
        "G": P.one_model([]),
        "두 단계: 여부 G+상대12": P.two_stage(P.REL_R),
        "두 단계: 여부 G+일치도2": P.two_stage(MATCH),
        "두 단계: 여부 G+상대12+일치도2": P.two_stage(P.REL_R + MATCH),
        "두 단계: 여부 G+sun_proxy6+일치도2": P.two_stage(V.SUN + MATCH),
    }
    scores = P.evaluate(data, a.repeats)
    a.out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(a.out / "scores.csv", index=False, encoding="utf-8-sig")
    cols = ["macro_f1", "alert_accuracy", "false_alert_rate", "alert_recall", "severe_precision", "severe_recall"]
    g = scores.groupby(["test", "method"], sort=False)[cols].mean().round(3)
    for t in ["group_cv", "held_camera:도로 광각", "held_camera:블랙박스", "held_camera:서울 폰", "reused_AIHub"]:
        print(f"\n== {t}")
        print(g.loc[t].to_string())


if __name__ == "__main__":
    main()
