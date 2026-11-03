"""두 단계 모델(여부 G+화면 → 심함 G)의 확률 보정 · 알림 기준 + 1단계 앙상블 비교.

확률은 전부 '카메라 하나 빼고 학습' 예측(처음 보는 카메라 = 앱 상황)으로 모은다.
    1단계 눈부심 여부  P(1+) — 후보: G+화면 로지스틱 / 앙상블(G 로지스틱 · G+화면 로지스틱 · G+화면 LightGBM 확률 평균)
    2단계 심함        P(2 | 눈부심) = G 로지스틱의 P(2) / (P(1) + P(2))
보는 것:
    ① 보정 — 구간별 예측 확률 vs 실제 비율 · Brier 점수 (원래 / 플랫 보정, 보정도 카메라 빼고 맞춤)
    ② 알림 기준 — 1단계는 재현율 ≥ 0.90 을 지키는 가장 높은 기준, 2단계는 심함 F1 이 가장 큰 기준.
       기준도 '두 카메라로 고르고 나머지 카메라로 채점' 해서 낙관을 뺌. BE 값은 세 카메라 전부로 고름
    ③ 기준을 적용한 3단계 점수 vs 지금(기준 0.5 · 0.5)

    python notebooks/experiments/glare_video_source/g_calibration_probe.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score

from nunchi_ml.glare.features import FEATURES
from nunchi_ml.glare.pilot_eval import lightgbm_model

import g_fusion_probe as F
import retrain_korea as R

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
RECALL_MIN = 0.90


def camera_out(d):
    """카메라 하나 빼고 학습한 예측 확률 (사진마다 한 번씩)."""
    out = pd.DataFrame(index=d.index, columns=["p_glare_fuse", "p_glare_ens", "p_sev"], dtype=float)
    for c in F.CAMS:
        te = d.camera.eq(c).to_numpy()
        tr, ts = d[~te], d[te]
        pg = F.proba(F.logit, tr, ts, F.G_COLS)
        pf = F.proba(F.logit, tr, ts, F.G_COLS + FEATURES)
        pl = F.proba(lightgbm_model, tr, ts, F.G_COLS + FEATURES)
        out.loc[te, "p_glare_fuse"] = pf[:, 1:].sum(1)
        out.loc[te, "p_glare_ens"] = (pg + pf + pl)[:, 1:].sum(1) / 3
        out.loc[te, "p_sev"] = pg[:, 2] / np.clip(pg[:, 1:].sum(1), 1e-9, None)
    return out


def platt(p, y, cam):
    """플랫 보정(로지스틱 한 겹)도 카메라 빼고: 다른 두 카메라로 맞춘 보정을 남은 카메라에."""
    q = np.zeros(len(p))
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, None)).reshape(-1, 1)
    for c in F.CAMS:
        te = cam == c
        m = LogisticRegression().fit(z[~te], y[~te])
        q[te] = m.predict_proba(z[te])[:, 1]
    return q


def reliability(p, y, bins=(0, .2, .4, .6, .8, 1.0001)):
    b = pd.cut(p, bins, right=False)
    t = pd.DataFrame({"구간": b, "p": p, "y": y}).groupby("구간", observed=True).agg(사진=("y", "size"), 예측=("p", "mean"), 실제=("y", "mean"))
    return " | ".join(f"{iv.left:.1f}~{min(iv.right, 1):.1f}: {r.사진}장 예측 {r.예측:.2f} 실제 {r.실제:.2f}" for iv, r in t.iterrows())


def pick_glare(p, y):
    """재현율 ≥ RECALL_MIN 을 지키는 가장 높은 기준 (없으면 가장 낮은 기준)."""
    ok = [t for t in GRID if (p[y == 1] >= t).mean() >= RECALL_MIN]
    return max(ok) if ok else GRID[0]


def pick_severe(p, y):
    """심함 F1 이 가장 큰 기준 (눈부심 정답 사진 안에서 1 vs 2)."""
    f1 = [(2 * ((p >= t) & y).sum() / max((p >= t).sum() + y.sum(), 1), t) for t in GRID]
    return max(f1)[1]


def three(pg, ps, t1, t2):
    return np.where(pg >= t1, np.where(ps >= t2, 2, 1), 0)


def main():
    d = F.data()
    y = d.label.to_numpy()
    cam = d.camera.to_numpy()
    glare, sev = (y >= 1).astype(int), y == 2
    o = camera_out(d)
    print(f"사진 {len(d)}장 · 눈부심 {glare.sum()} · 심함 {sev.sum()} (카메라 하나 빼고 예측)\n")

    print("① 보정 — 1단계 P(눈부심)")
    for name, col in (("G+화면 로지스틱", "p_glare_fuse"), ("앙상블 (3개 평균)", "p_glare_ens")):
        p = o[col].to_numpy()
        q = platt(p, glare, cam)
        print(f"  {name:16s} Brier 원래 {brier_score_loss(glare, p):.3f} → 플랫 {brier_score_loss(glare, q):.3f}")
        print(f"    원래  {reliability(p, glare)}")
        print(f"    플랫  {reliability(q, glare)}")
    m = glare == 1
    ps = o.p_sev.to_numpy()
    print(f"  2단계 P(심함 | 눈부심)  Brier {brier_score_loss(sev[m], ps[m]):.3f}")
    print(f"    {reliability(ps[m], sev[m].astype(int))}\n")

    print(f"② 알림 기준 — 두 카메라로 고르고 남은 카메라로 채점 (1단계 재현율 ≥ {RECALL_MIN:.2f} 목표)")
    for name, col in (("G+화면 로지스틱", "p_glare_fuse"), ("앙상블 (3개 평균)", "p_glare_ens")):
        p = o[col].to_numpy()
        pred = np.zeros(len(d), int)
        for c in F.CAMS:
            te = cam == c
            t1 = pick_glare(p[~te], glare[~te])
            t2 = pick_severe(ps[~te & m], sev[~te & m])
            pred[te] = three(p[te], ps[te], t1, t2)
            r = (p[te][glare[te] == 1] >= t1).mean()
            fp = (p[te][glare[te] == 0] >= t1).mean() if (glare[te] == 0).any() else np.nan
            print(f"  {name:16s} → {c:5s} 기준 {t1:.2f} · {t2:.2f} | 여부 재현율 {r:.2f} · 정상을 눈부심이라 한 비율 {fp:.2f}")
        base = three(p, ps, 0.5, 0.5)
        print(f"    3단계  지금(0.5 · 0.5) {F.fmt(R.scores(y, base))}  →  고른 기준 {F.fmt(R.scores(y, pred))}")
        for n, q in (("지금", base), ("고른 기준", pred)):
            dist = " · ".join(f"{c} {np.bincount(q[cam == c], minlength=3).tolist()}" for c in F.CAMS)
            print(f"    {n:6s} 심함 정밀도 {precision_score(y, q, labels=[2], average='macro', zero_division=0):.2f}"
                  f" | 예측 [0, 1, 2] 장수 {dist}")
    print("    (정답 " + " · ".join(f"{c} {np.bincount(y[cam == c], minlength=3).tolist()}" for c in F.CAMS) + ")")

    p = o.p_glare_fuse.to_numpy()
    t1, t2 = pick_glare(p, glare), pick_severe(ps[m], sev[m])
    print(f"\n③ BE 에 넣을 기준 (세 카메라 전부로 고름, G+화면 로지스틱): 눈부심 알림 P(1+) ≥ {t1:.2f} · 강한 경고 P(2|눈부심) ≥ {t2:.2f}")
    print(f"   이 기준의 3단계 점수(참고, 고른 데이터로 채점이라 낙관적) {F.fmt(R.scores(y, three(p, ps, t1, t2)))}")


if __name__ == "__main__":
    main()
