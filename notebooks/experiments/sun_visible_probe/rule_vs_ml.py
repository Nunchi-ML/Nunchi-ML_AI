"""규칙 vs ML — 공정한 비교 (모델 ②). 규칙에게도 학습 데이터로 경계값을 튜닝할 기회를 준다.

**결과를 보기 전에 고정한 것**
  주 지표 3개: ① 세 카메라 촬영 단위 CV 5×30 의 macro F1  ② 카메라 하나 빼고 시험(세 카메라 합산) 눈부심 알림정확도
              ③ 재사용 AI Hub 알림정확도.  보조: 헛알림 · 알림재현율 · 심함 정밀도/재현율 (전부 보고).
  비교 대상:
    규칙 A  팀 기준 손 규칙        해 높이 0~25° · Δ ≤ 25° · 구름 ≤ 7 → Δ ≤ 15° 면 심함 (고정, 학습 없음)
    규칙 B  경계값을 학습 데이터로 고른 규칙   같은 모양, 경계값 4개를 학습 fold 에서만 격자 탐색 (macro F1 최대)
    규칙 C  물리 공식 규칙         Stiles-Holladay 가림 휘도 Lv = 10·E/θ² (θ = 시선-해 각도(°),
                                  E = 1000·sin(고도)·(1 − 0.75·(oktas/8)^3.4), oktas = 구름×0.8) 의 문턱 2개를 학습 fold 에서 고름
    ML 1    로지스틱 G (최종 규칙: P(눈부심) ≥ .5, P(심함|눈부심) ≥ .5)
    ML 2    로지스틱 G + 물리 지수 log10 Lv            ("물리 + ML")
    ML 3    LightGBM G (단조 제약)
    ML 4    두 단계 G + 사진 상대 특징 12
  프로토콜: Codex g_visibility_probe 와 같은 고정 스냅샷 · previous 날씨 · 같은 판정 함수.
  통계: AI Hub 는 주행, 카메라 빼고 시험은 촬영묶음 단위 군집 부트스트랩 (3000회).
  조건을 보고 이기는 지표를 고르지 않는다 — 위 지표를 전부 출력한다.

    python notebooks/experiments/sun_visible_probe/rule_vs_ml.py --snapshot ... --rel ... --cache ... --out ...
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lightgbm as lgb
import numpy as np
import pandas as pd

import codex_protocol_compare as P
import g_visibility_probe as C
from g_weather_time_probe import G_COLUMNS, replay
from rel_features import REL

MONO = {"sun_alt": 0, "u": 0, "v": 0, "in_fov": 1, "cloud": -1, "sunshine": 1}


def lv_score(d):
    """log10 Stiles-Holladay 가림 휘도 (해가 지평선 아래면 매우 작은 값)."""
    alt = np.radians(d.sun_alt.clip(lower=-90, upper=90).to_numpy(float))
    cos_t = np.cos(alt) * np.cos(np.radians(d.gap.to_numpy(float)))
    theta = np.degrees(np.arccos(np.clip(cos_t, -1, 1))).clip(min=1.0)
    okta = d.cloud.clip(0, 10).to_numpy(float) * 0.8
    e = 1000 * np.sin(np.clip(alt, 0, None)) * (1 - 0.75 * (okta / 8) ** 3.4)
    return np.log10(10 * e / theta ** 2 + 1e-3)


def macro_f1(y, p):
    f = []
    for c in (0, 1, 2):
        tp = ((y == c) & (p == c)).sum(); fp = ((y != c) & (p == c)).sum(); fn = ((y == c) & (p != c)).sum()
        f.append(0. if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def onehot(pred):
    p = np.zeros((len(pred), 3)); p[np.arange(len(pred)), pred] = 1.
    return p


def rule_hand(tr, te):
    on = (te.sun_alt > 0) & (te.sun_alt <= 25) & (te.gap <= 25) & (te.cloud <= 7)
    return onehot(np.where(on & (te.gap <= 15), 2, np.where(on, 1, 0)).astype(int))


def rule_tuned(tr, te):
    alt, gap, cl, y = (tr[c].to_numpy(float) for c in ("sun_alt", "gap", "cloud", "label"))
    y = y.astype(int); best, arg = -1, None
    for a in range(15, 50, 5):
        for g in range(15, 65, 5):
            for c in range(4, 10):
                on = (alt > 0) & (alt <= a) & (gap <= g) & (cl <= c)
                for s in range(5, 30, 5):
                    if s > g:
                        continue
                    f = macro_f1(y, np.where(on & (gap <= s), 2, np.where(on, 1, 0)))
                    if f > best:
                        best, arg = f, (a, g, c, s)
    a, g, c, s = arg
    on = (te.sun_alt > 0) & (te.sun_alt <= a) & (te.gap <= g) & (te.cloud <= c)
    return onehot(np.where(on & (te.gap <= s), 2, np.where(on, 1, 0)).astype(int))


def rule_physics(tr, te):
    s, y = tr.lv.to_numpy(), tr.label.to_numpy().astype(int)
    qs = np.unique(np.quantile(s, np.linspace(.05, .95, 37))); best, arg = -1, (qs[0], qs[-1])
    for i, t1 in enumerate(qs):
        for t2 in qs[i:]:
            f = macro_f1(y, np.where(s >= t2, 2, np.where(s >= t1, 1, 0)))
            if f > best:
                best, arg = f, (t1, t2)
    t1, t2 = arg
    return onehot(np.where(te.lv >= t2, 2, np.where(te.lv >= t1, 1, 0)).astype(int))


def lgbm_g(tr, te):
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=7, min_child_samples=5, class_weight="balanced",
                           monotone_constraints=[MONO[c] for c in G_COLUMNS], random_state=0, n_jobs=1, verbose=-1)
    m.fit(tr[G_COLUMNS], tr.label)
    p = np.zeros((len(te), 3)); p[:, m.classes_] = m.predict_proba(te[G_COLUMNS]); return p


METHODS = {
    "규칙 A 팀 기준 손 규칙": rule_hand,
    "규칙 B 경계값 학습 규칙": rule_tuned,
    "규칙 C 물리 공식 규칙": rule_physics,
    "ML 1 로지스틱 G (최종)": P.one_model([]),
    "ML 2 로지스틱 G + 물리지수": P.one_model(["lv"]),
    "ML 3 LightGBM G (단조)": lgbm_g,
    "ML 4 두 단계 G+상대12": P.two_stage(P.REL_R),
}


def boot(preds, y, clusters, fn, reps=3000, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(clusters); idx = {c: np.where(clusters == c)[0] for c in u}
    out = {k: [] for k in preds}
    for _ in range(reps):
        ii = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
        for k, p in preds.items():
            out[k].append(fn(y[ii], p[ii]))
    return {k: np.array(v) for k, v in out.items()}


def alert_acc(y, p):
    return float(np.mean((y >= 1) == (p >= 1)))


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
    data = data.merge(rel, on="id", how="left", validate="one_to_one")
    data["lv"] = lv_score(data)
    P.METHODS = METHODS
    scores = P.evaluate(data, a.repeats)
    a.out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(a.out / "scores.csv", index=False, encoding="utf-8-sig")
    cols = ["macro_f1", "alert_accuracy", "false_alert_rate", "alert_recall", "severe_precision", "severe_recall"]
    g = scores.groupby(["test", "method"], sort=False)[cols].mean().round(3)
    for t in ["group_cv", "held_camera:도로 광각", "held_camera:블랙박스", "held_camera:서울 폰", "reused_AIHub"]:
        print(f"\n== {t}"); print(g.loc[t].to_string())

    train = data[data.camera.ne("AI Hub")].reset_index(drop=True)
    test = data[data.camera.eq("AI Hub")].reset_index(drop=True)
    names = list(METHODS)
    # 카메라 빼고 시험 (세 카메라 합산) 군집 부트스트랩
    pred_l = {n: np.zeros(len(train), int) for n in names}
    for cam in train.camera.unique():
        h = train.camera.eq(cam).to_numpy()
        for n, fn in METHODS.items():
            pred_l[n][h] = C.decision(fn(train[~h], train[h]))
    yl = train.label.to_numpy().astype(int)
    print("\n== 카메라 하나 빼고 시험 (세 카메라 합산) 눈부심 알림정확도 · 군집(촬영묶음) 부트스트랩 95%")
    b = boot(pred_l, yl, train.group.to_numpy(), alert_acc)
    for n in names:
        print(f"  {n:26s} {alert_acc(yl, pred_l[n]):.3f}  [{np.percentile(b[n], 2.5):.3f}, {np.percentile(b[n], 97.5):.3f}]")
    # 재사용 AI Hub
    ya = test.label.to_numpy().astype(int)
    pred_a = {n: C.decision(fn(train, test)) for n, fn in METHODS.items()}
    print("\n== 재사용 AI Hub 눈부심 알림정확도 · 주행 군집 부트스트랩 95%")
    b2 = boot(pred_a, ya, (test.area + "/" + test.drive_name).to_numpy(), alert_acc)
    for n in names:
        print(f"  {n:26s} {alert_acc(ya, pred_a[n]):.3f}  [{np.percentile(b2[n], 2.5):.3f}, {np.percentile(b2[n], 97.5):.3f}]")
    base = "ML 1 로지스틱 G (최종)"
    for label, bb in (("카메라 빼고", b), ("재사용 AI Hub", b2)):
        print(f"\n  [{label}] 차이 (행 − {base}), 95% 구간")
        for n in names:
            if n != base:
                d = bb[n] - bb[base]; print(f"    {n:26s} {d.mean():+.3f}  [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")


if __name__ == "__main__":
    main()
