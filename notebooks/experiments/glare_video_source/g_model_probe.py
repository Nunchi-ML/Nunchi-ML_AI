"""G (해 위치 + 날씨) — 손 규칙 vs 데이터로 학습한 모델. 화면 특징 없이 카메라와 상관없는 값만.

입력 (라벨 사진 중 진행 방향을 믿을 수 있는 것):
    대구  블랙박스 · 도로 광각 (성진 라벨, ASOS 대구 143)
    서울  운전 시점 60장 (3인 다수결, ASOS 서울 108, make_seoul_label_set.py · compare_aihub_labels.py --dir data/glare_mapillary_seoul)
    AI Hub 역광 304장 (G_AIHUB=1 일 때만 · 3인 다수결 · aihub_heading.py 로 복원한 주행 방향 · ASOS 서울 108 · 대전 133 · 창원 155)
    운전자 기준  sun_alt · gap(Δ) · cloud · sunshine                    — 앱이 GPS · 나침반 · 시각 · 날씨만 알면 됨
    화각 반영    sun_alt · u · v · in_fov · cloud · sunshine            — 카메라 화각까지 알 때
모델:
    규칙         해 높이 0~25° · Δ ≤ 25° · 전운량 ≤ 7 → Δ ≤ 15° 면 2, 아니면 1, 그 밖 0 (지금 기준을 그대로)
    로지스틱     다항 로지스틱 (표준화) — 계수로 설명
    결정트리     깊이 3 — 학습된 경계가 규칙 모양으로 나옴
    LightGBM     단조 제약 (Δ · 구름 ↑ → 눈부심 ↓, 일조 ↑ → 눈부심 ↑)
검증:
    ① 카메라 하나 빼고 학습 → 뺀 카메라로 시험 (블랙박스 · 도로 광각 · 서울 폰)
    ② 촬영 단위 GroupKFold 5 × 30 반복 (전부 섞어서)

    python notebooks/experiments/glare_video_source/g_model_probe.py
    G_AIHUB=1 python notebooks/experiments/glare_video_source/g_model_probe.py     # AI Hub 를 네 번째 카메라로 (②-v4)
"""
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import fetch_asos as A
import find_sun_frames as F
import fov_probe as P
import retrain_korea as R
import v2_candidate as V

OUT = F.OUT
SEOUL = F.ROOT / "data" / "glare_mapillary_seoul"
AIHUB_DIR = F.ROOT / "data" / "aihub_special"
AIHUB = os.getenv("G_AIHUB") == "1"
AIHUB_STN = {"seoul2": 108, "daejeon": 133, "changwon": 155}
AIHUB_FOCAL, AIHUB_W, AIHUB_H = 1067.25 / 1920, 1920, 1200   # calib.txt fx (세 지역 같음)
SEED = 0
SETS = {"운전자 기준": ["sun_alt", "gap", "cloud", "sunshine"],
        "화각 반영": ["sun_alt", "u", "v", "in_fov", "cloud", "sunshine"]}
MONO = {"sun_alt": 0, "gap": -1, "cloud": -1, "sunshine": 1, "u": 0, "v": 0, "in_fov": 1}


def data(*, weather_alignment="previous"):
    k = V.korea_with_sun()
    t = pd.concat([pd.read_csv(OUT / n)[["id", "captured_at_kst"]] for n in ("labels_drive.csv", "labels_dash.csv")]).drop_duplicates("id")
    k = A.attach(k.merge(t, on="id"), alignment=weather_alignment)
    k = k[k.sun_alt.notna() & k.gap.notna() & k.cloud.notna()].reset_index(drop=True)
    k = pd.concat([k, seoul(weather_alignment=weather_alignment)]
                  + ([aihub(weather_alignment=weather_alignment)] if AIHUB else []), ignore_index=True)
    k[["u", "v", "in_fov"]] = k[["u", "v", "in_fov"]].fillna({"u": 9, "v": -9, "in_fov": 0})   # 해가 뒤쪽 → 화면 밖
    return k


def seoul(*, weather_alignment="previous"):
    m = pd.read_csv(SEOUL / "labels_merged.csv")[["id", "majority"]].dropna()
    s = pd.read_csv(SEOUL / "label_set.csv").merge(m, on="id")
    s = s.merge(P.camera_params(s.id.tolist(), SEOUL)[["id", "focal", "width", "height"]], on="id")
    s["heading_suspect"] = False                                   # 세트를 만들 때 이미 뺌
    s = pd.concat([s.reset_index(drop=True), P.fov_features(s).reset_index(drop=True)], axis=1)
    s = A.attach(s, stn=108, alignment=weather_alignment)
    s = s.assign(label=s.majority.astype(int), camera="서울 폰",
                 group="se:" + s.model.fillna("?").astype(str) + " " + s.captured_at_kst.str[:10])
    return s[s.focal.notna() & s.cloud.notna()]


def aihub(*, weather_alignment="previous"):
    m = pd.read_csv(AIHUB_DIR / "labels_merged.csv")[["id", "file", "sequence", "majority"]].dropna()
    m["area"], m["drive_name"] = m.sequence.str.split("/").str[0], m.sequence.str.split("/").str[1]
    h = pd.read_csv(AIHUB_DIR / "drive_heading.csv")
    h["drive_name"] = h.drive.str.split("/").str[-1]
    s = m.merge(h[h.split.eq("Validation")][["area", "drive_name", "time", "lat", "lon", "heading", "sun_alt", "sun_az", "gap"]],
                on=["area", "drive_name"])
    s = s.merge(pd.read_csv(AIHUB_DIR / "aihub_features.csv"), on="id")
    s = s.assign(focal=AIHUB_FOCAL, width=AIHUB_W, height=AIHUB_H, heading_suspect=False, captured_at_kst=s.time.str[:19])
    s = pd.concat([s.reset_index(drop=True), P.fov_features(s).reset_index(drop=True)], axis=1)
    s = pd.concat([A.attach(g, stn=AIHUB_STN[a], alignment=weather_alignment)
                   for a, g in s.groupby("area")], ignore_index=True)
    s = s.assign(label=s.majority.astype(int), camera="AI Hub", group="ah:" + s.area + " " + s.captured_at_kst.str[:10])
    return s[s.heading.notna() & s.cloud.notna()]


def rule(d):
    on = (d.sun_alt > 0) & (d.sun_alt <= 25) & (d.gap <= 25) & (d.cloud <= 7)
    return np.where(on & (d.gap <= 15), 2, np.where(on, 1, 0))


MODELS = {
    "로지스틱": lambda cols: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "결정트리 깊이 3": lambda cols: DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=SEED),
    "LightGBM 단조": lambda cols: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=5,
                                                   class_weight="balanced", monotone_constraints=[MONO[c] for c in cols],
                                                   random_state=SEED, n_jobs=1, verbose=-1),
}


def fmt(s):
    return f"F1 {s['macro F1']:.2f} · 심함 {s['심함 재현율']:.2f} · 여부 {s['눈부심 여부 정확도']:.2f}"


def main():
    k = data(weather_alignment=os.getenv("G_WEATHER_ALIGNMENT", "previous"))
    print("Weather alignment:", k.weather_time_policy.unique().tolist(),
          "| publication availability verified:", int(k.weather_availability_verified.sum()), "/", len(k))
    y = k.label.to_numpy()
    cams = tuple(k.camera.unique())
    print(f"사진 {len(k)}장 · " + " · ".join(f"{c} {int(k.camera.eq(c).sum())}" for c in cams)
          + f" · 라벨 {k.label.value_counts().sort_index().to_dict()} · 촬영 {k.group.nunique()}개\n")

    print("① 카메라 하나 빼고 학습 → 뺀 카메라로 시험")
    p = rule(k)
    print(f"  {'규칙 (학습 없음)':24s} " + " | ".join(f"→{c} {fmt(R.scores(y[k.camera.eq(c)], p[k.camera.eq(c)]))}" for c in cams))
    for sname, cols in SETS.items():
        for mname, make in MODELS.items():
            out = []
            for te_c in cams:
                te = k.camera.eq(te_c).to_numpy()
                pp = make(cols).fit(k.loc[~te, cols], y[~te]).predict(k.loc[te, cols])
                out.append(f"→{te_c} {fmt(R.scores(y[te], pp))}")
            print(f"  {sname + ' · ' + mname:24s} " + " | ".join(out))

    print("\n② 촬영 단위 GroupKFold 5 × 30 (전부 섞음, 평균 ± 표준편차)")
    print(f"  {'규칙 (학습 없음)':24s} {fmt(R.scores(y, rule(k)))}")
    for sname, cols in SETS.items():
        for mname, make in MODELS.items():
            res = []
            for r in range(30):
                pp = np.zeros(len(k), int)
                for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(k, y, k.group):
                    pp[te] = make(cols).fit(k.loc[tr, cols], y[tr]).predict(k.loc[te, cols])
                res.append(R.scores(y, pp))
            s = pd.DataFrame(res)
            print(f"  {sname + ' · ' + mname:24s} " + " · ".join(
                f"{n} {s[c].mean():.2f}±{s[c].std():.2f}" for n, c in (("F1", "macro F1"), ("심함", "심함 재현율"), ("여부", "눈부심 여부 정확도"))))

    cols = SETS["운전자 기준"]
    tree = MODELS["결정트리 깊이 3"](cols).fit(k[cols], y)
    print("\n전체로 학습한 결정트리 (운전자 기준) — 데이터가 정한 경계\n" + export_text(tree, feature_names=cols, decimals=1))
    lr = MODELS["로지스틱"](cols).fit(k[cols], y)
    coef = pd.DataFrame(lr[-1].coef_, columns=cols, index=["0 정상", "1 눈부심", "2 심함"]).round(2)
    print("로지스틱 계수 (표준화 단위, 클래스별)\n" + coef.to_string())


if __name__ == "__main__":
    main()
