"""합성 글레어 증강 — 해가 없는 정상 사진에 해 원반 · 번짐 · 안개 베일을 합성해 눈부심 · 심함 예시를 늘리면 나아지나.

규칙 (결과를 좋게 만들려고 고르지 않는다 — 전부 사전에 정함):
  * 합성 원본은 **학습 쪽 카메라의 정상(0) 사진만**. AI Hub 는 원본으로 쓰지 않고 시험으로만 둔다.
  * 접힌(fold) 학습 사진에서 나온 합성만 그 fold 학습에 쓴다 (원본 id 로 걸러 촬영 누수 없음).
  * 합성 종류 (원본 1장당 2개씩): 노출만 흔든 정상(0) · 약한 글레어(1) · 강한 글레어+안개 베일(2).
    '노출만 흔든 정상' 은 합성 흔적 자체를 눈부심 신호로 배우지 않게 하는 대조.
  * 합성 행의 G 입력은 학습 fold 의 같은 클래스 실제 행에서 무작위로 가져옴 (이미지와 G 의 짝을 만들지 않음).
  * 합성 가중치 0.5 (고정, 튜닝 안 함). 시험은 실제 사진만.
비교 (Codex 프로토콜 · 같은 판정 함수):
    두 단계 G+상대12                 실제만 (기준)
    + 노출 흔든 정상만                합성 정상만 더함 (대조)
    + 합성 (여부만)                   여부 단계에만 합성, 심함은 G 실제
    + 합성 (여부 · 심함 모두 이미지)   하나의 G+상대12 로지스틱을 합성 포함 학습 (.5/.5)
    심함도 이미지 쓰는 실제만 (대조)    합성 없이 같은 구조

    python notebooks/experiments/sun_visible_probe/synth_probe.py --snapshot ... --rel ... --cache ... --out ...
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import codex_protocol_compare as P
import g_visibility_probe as C
from g_weather_time_probe import G_COLUMNS, replay
from rel_features import REL, rel_array

W_SYN = 0.5
PER_KIND = 2


def add_glare(im, rng, level):
    """im: BGR uint8 640×368. level 0 = 노출만 흔듦, 1 = 약한 글레어, 2 = 강한 글레어 + 베일."""
    h, w = im.shape[:2]
    x = (im.astype(np.float32) / 255.) ** rng.uniform(.8, 1.2) * rng.uniform(.75, 1.25)
    x = np.clip(x, 0, 1)
    if level == 0:
        return (x * 255).astype(np.uint8)
    cx, cy = rng.uniform(.1, .9) * w, rng.uniform(.05, .5) * h
    yy, xx = np.mgrid[:h, :w]
    d = np.hypot(xx - cx, yy - cy)
    r = w * rng.uniform(.012, .025)
    sigma, a = (w * rng.uniform(.04, .09), rng.uniform(.35, .6)) if level == 1 else (w * rng.uniform(.10, .22), rng.uniform(.7, 1.))
    disc = 1. / (1. + np.exp((d - r) / (.15 * r)))
    glow = np.clip(disc + a * np.exp(-d ** 2 / (2 * sigma ** 2)), 0, 1)[..., None]
    out = 1 - (1 - x) * (1 - glow * np.array([.88, .96, 1.], np.float32))
    if level == 2:
        veil = (rng.uniform(.25, .5) * np.exp(-d ** 2 / (2 * (.6 * w) ** 2)))[..., None]
        out = out * (1 - veil) + veil
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def build_pool(data, paths, seed=0, sheet=None):
    rng = np.random.default_rng(seed)
    src = data[(data.camera.ne("AI Hub")) & (data.label == 0)]
    rows, tiles = [], []
    for r in src.itertuples():
        im = cv2.imdecode(np.fromfile(str(paths[r.id]), np.uint8), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (640, 368), interpolation=cv2.INTER_LINEAR)
        for level in (0, 1, 2):
            for k in range(PER_KIND):
                s = add_glare(im, rng, level)
                rows.append({"src_id": r.id, "level": level, **{"r_" + c: v for c, v in rel_array(s).items()}})
                if sheet is not None and k == 0 and len(tiles) < 24 and r.Index % 4 == 0:
                    tiles.append(cv2.resize(s, (320, 184)))
    if sheet is not None and tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
        cv2.imencode(".jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))[1].tofile(str(sheet))
    return pd.DataFrame(rows)


def syn_rows(train, pool, levels, seed):
    """학습 fold 에 해당하는 합성 행 (G 입력은 같은 클래스 실제 행에서 무작위)."""
    s = pool[pool.src_id.isin(set(train.id)) & pool.level.isin(levels)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    g = []
    for lvl in s.level:
        c = train[train.label == lvl]
        g.append(c.iloc[rng.integers(len(c))] if len(c) else train[train.label >= 1].iloc[rng.integers((train.label >= 1).sum())])
    g = pd.DataFrame(g)[G_COLUMNS].reset_index(drop=True)
    return pd.concat([g, s[P.REL_R], pd.Series(s.level, name="label")], axis=1)


def fit_proba(train, test, cols, extra=None):
    X, y, w = train[cols], train.label.to_numpy(), np.ones(len(train))
    if extra is not None and len(extra):
        X, y, w = pd.concat([X, extra[cols]]), np.r_[y, extra.label.to_numpy()], np.r_[w, np.full(len(extra), W_SYN)]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    m.fit(X, y, logisticregression__sample_weight=w)
    p = np.zeros((len(test), 3))
    p[:, m.classes_] = m.predict_proba(test[cols])
    return p


def combine(g, s):
    return np.c_[1 - g, g * (1 - s), g * s]


def make_methods(pool):
    GR = G_COLUMNS + P.REL_R

    def fold_seed(train):
        return int(pd.util.hash_pandas_object(train.id, index=False).sum() % (2 ** 31))

    def base(tr, te):
        pf, pg = fit_proba(tr, te, GR), fit_proba(tr, te, G_COLUMNS)
        return combine(pf[:, 1:].sum(1), pg[:, 2] / np.maximum(pg[:, 1] + pg[:, 2], 1e-9))

    def jitter_only(tr, te):
        pf = fit_proba(tr, te, GR, syn_rows(tr, pool, [0], fold_seed(tr)))
        pg = fit_proba(tr, te, G_COLUMNS)
        return combine(pf[:, 1:].sum(1), pg[:, 2] / np.maximum(pg[:, 1] + pg[:, 2], 1e-9))

    def syn_stage1(tr, te):
        pf = fit_proba(tr, te, GR, syn_rows(tr, pool, [0, 1, 2], fold_seed(tr)))
        pg = fit_proba(tr, te, G_COLUMNS)
        return combine(pf[:, 1:].sum(1), pg[:, 2] / np.maximum(pg[:, 1] + pg[:, 2], 1e-9))

    def syn_both(tr, te):
        return fit_proba(tr, te, GR, syn_rows(tr, pool, [0, 1, 2], fold_seed(tr)))

    def real_both(tr, te):
        return fit_proba(tr, te, GR)

    return {"두 단계 G+상대12 (실제만, 기준)": base,
            "+ 노출 흔든 정상만 (대조)": jitter_only,
            "+ 합성 (여부만)": syn_stage1,
            "+ 합성 (여부·심함 모두 이미지)": syn_both,
            "한 모델 G+상대12 (합성 없이, 대조)": real_both}


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
    paths = dict(zip(snap.id, C.image_paths(snap, a.data_root)))
    data = C.features(snap, list(paths.values()), a.cache)
    rel = pd.read_csv(a.rel, dtype={"id": str})[["id"] + REL].rename(columns={c: "r_" + c for c in REL})
    data = data.merge(rel, on="id", how="left", validate="one_to_one")
    a.out.mkdir(parents=True, exist_ok=True)
    pool = build_pool(data, paths, sheet=a.out / "synthetic_examples.jpg")
    print(f"합성 풀 {len(pool)}장 (원본 정상 {pool.src_id.nunique()}장 × 3종 × {PER_KIND})")
    P.METHODS = make_methods(pool)
    scores = P.evaluate(data, a.repeats)
    scores.to_csv(a.out / "scores.csv", index=False, encoding="utf-8-sig")
    cols = ["macro_f1", "alert_accuracy", "false_alert_rate", "alert_recall", "severe_precision", "severe_recall"]
    g = scores.groupby(["test", "method"], sort=False)[cols].mean().round(3)
    for t in ["group_cv", "held_camera:도로 광각", "held_camera:블랙박스", "held_camera:서울 폰", "reused_AIHub"]:
        print(f"\n== {t}")
        print(g.loc[t].to_string())


if __name__ == "__main__":
    main()
