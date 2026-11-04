"""Codex 프로토콜(g_visibility_probe, 고정 스냅샷 · previous 날씨 · 같은 판정 함수 · 촬영 CV 5×30 · 카메라 제외 · 재사용 AI Hub)에
'두 단계 구조' 와 sun_visible_probe 의 상대 특징 12개를 행으로 더해 비교한다. Codex 파일은 수정하지 않고 import 만 한다.

두 단계 = 눈부심 여부는 G + 이미지 특징 로지스틱의 P(눈부심) ≥ .5, 심함은 G 만 로지스틱의 P(심함 | 눈부심) ≥ .5.
판정 함수는 g_visibility_probe.decision 을 그대로 쓴다 (두 단계를 확률 3칸으로 환산해 같은 함수에 넣음).

    python notebooks/experiments/sun_visible_probe/codex_protocol_compare.py --snapshot data/analysis_20261104/model2_snapshot.csv \
        --rel data/sun_visible/rel_features.csv --cache <image_features.csv> --out <폴더>
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "glare_video_source"))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import g_visibility_probe as C
from g_weather_time_probe import G_COLUMNS, replay
from nunchi_ml.glare import visibility as V
from rel_features import REL

SUN = V.SUN
RING3 = ["r_ring3"]
REL_R = ["r_" + c for c in REL]   # 이름 충돌(peak_x · peak_y) 방지


def one_model(cols):
    return lambda tr, te: C.probabilities(tr, te, G_COLUMNS + cols)


def two_stage(cols):
    def f(tr, te):
        pf = C.probabilities(tr, te, G_COLUMNS + cols)
        pg = C.probabilities(tr, te, G_COLUMNS)
        g = pf[:, 1:].sum(1)
        s = pg[:, 2] / np.maximum(pg[:, 1] + pg[:, 2], 1e-9)
        return np.c_[1 - g, g * (1 - s), g * s]
    return f


METHODS = {
    "G": one_model([]),
    "한 모델: G+sun_proxy6 (Codex)": one_model(SUN),
    "두 단계: 여부 G+sun_proxy6": two_stage(SUN),
    "두 단계: 여부 G+sun_proxy6+ring3": two_stage(SUN + RING3),
    "한 모델: G+상대12": one_model(REL_R),
    "두 단계: 여부 G+상대12": two_stage(REL_R),
}


def evaluate(data, repeats):
    train = data[data.camera.ne("AI Hub")].reset_index(drop=True)
    test = data[data.camera.eq("AI Hub")].reset_index(drop=True)
    rows = []

    def rec(method, kind, repeat, subset, p):
        rows.append(dict(method=method, test=kind, repeat=repeat, **C.score(subset.label, C.decision(p))))

    for name, fn in METHODS.items():
        rec(name, "reused_AIHub", -1, test, fn(train, test))
        for cam in train.camera.unique():
            h = train.camera.eq(cam)
            rec(name, f"held_camera:{cam}", -1, train[h], fn(train[~h], train[h]))
        for seed in range(repeats):
            p = np.zeros((len(train), 3))
            for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(train, train.label, train.group):
                p[te] = fn(train.iloc[tr], train.iloc[te])
            rec(name, "group_cv", seed, train, p)
    return pd.DataFrame(rows)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--snapshot", type=Path, required=True)
    a.add_argument("--rel", type=Path, required=True)
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--data-root", type=Path, default=Path("data"))
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--repeats", type=int, default=30)
    a = a.parse_args()
    snap = pd.read_csv(a.snapshot, dtype={"id": str})
    snap = replay(snap, a.data_root / "asos", "previous")
    data = C.features(snap, C.image_paths(snap, a.data_root), a.cache)
    rel = pd.read_csv(a.rel, dtype={"id": str})[["id"] + REL].rename(columns={c: "r_" + c for c in REL})
    data = data.merge(rel, on="id", how="left", validate="one_to_one")
    if data[REL_R].isna().any().any():
        raise ValueError("상대 특징 결측")
    scores = evaluate(data, a.repeats)
    a.out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(a.out / "scores.csv", index=False, encoding="utf-8-sig")
    cols = ["macro_f1", "alert_accuracy", "false_alert_rate", "alert_recall", "severe_precision", "severe_recall"]
    g = scores.groupby(["test", "method"], sort=False)[cols].mean().round(3)
    for t in ["group_cv", "held_camera:도로 광각", "held_camera:블랙박스", "held_camera:서울 폰", "reused_AIHub"]:
        print(f"\n== {t}")
        print(g.loc[t].to_string())


if __name__ == "__main__":
    main()
