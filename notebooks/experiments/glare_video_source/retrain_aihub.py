"""모델 ② 재학습 실험 (AI Hub 역광 304장 · 3인 다수결 라벨 추가).

① AI Hub 카메라 안에서만: 반복 GroupKFold(5, 30번)
   - 주행 단위 그룹 (엄격 — 심한 눈부심은 거의 drive_1189 한 주행이라 시험 폴드에 가면 학습에 없음)
   - 주행 안 연속 10장 묶음 그룹 (같은 카메라 · 같은 주행이 섞임 → 낙관적, "같은 카메라 안" 상한)
② 카메라 하나 빼고 학습 → 뺀 카메라로 시험 (GLARE · 블랙박스 · 도로 광각 · AI Hub)
   GLARE 만 학습(지금 배포 방식)과 비교. 모델 · 특징 12개는 배포와 같음 (pilot_eval.lightgbm_model).

    python notebooks/experiments/glare_video_source/retrain_aihub.py --glare ../../data/glare/pilot_glare
결과: data/aihub_special/aihub_features.csv (사진별 화면 특징 12개, 다시 안 뽑음)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from nunchi_ml.glare.features import FEATURES, extract
from nunchi_ml.glare.pilot_eval import load, lightgbm_model

from retrain_korea import korea, scores

OUT = Path(__file__).resolve().parents[3] / "data" / "aihub_special"
BLOCK = 10


def aihub():
    d = pd.read_csv(OUT / "labels_merged.csv")
    d = d[d.majority.notna()].assign(label=lambda x: x.majority.astype(int))
    cache = OUT / "aihub_features.csv"
    if cache.exists():
        f = pd.read_csv(cache)
    else:
        f = pd.DataFrame([{"id": i, **extract(str(OUT / p))} for i, p in zip(d.id, d.file)])[["id"] + FEATURES]
        f.to_csv(cache, index=False, encoding="utf-8-sig")
    d = d.merge(f, on="id")
    d["block"] = d.sequence + ":" + (d.groupby("sequence").cumcount() // BLOCK).astype(str)
    return d.assign(group="ah:" + d.sequence, camera="AI Hub").reset_index(drop=True)


def repeated(d, groups, repeats=30):
    y, out = d.label.to_numpy(), []
    for r in range(repeats):
        p = np.zeros(len(d), int)
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(d, y, groups):
            p[te] = lightgbm_model().fit(d.iloc[tr][FEATURES], y[tr]).predict(d.iloc[te][FEATURES])
        out.append(scores(y, p))
    s = pd.DataFrame(out)
    return {k: f"{s[k].mean():.2f} ± {s[k].std():.2f}" for k in ("macro F1", "심함 재현율", "눈부심 여부 정확도")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glare", required=True)
    a = ap.parse_args()
    g = load(Path(a.glare) / "glare_features.csv", Path(a.glare) / "pilot_labels.csv")
    g = g.assign(group="us:" + g.vid, camera="GLARE")[["group", "camera", "label"] + FEATURES]
    k = korea()[["group", "camera", "label"] + FEATURES]
    h = aihub()
    print(f"AI Hub {len(h)}장 · 라벨 {h.label.value_counts().sort_index().to_dict()} · 주행 {h.sequence.nunique()}개\n")

    print("① AI Hub 카메라 안에서만 (반복 GroupKFold 5 × 30)")
    for name, grp in (("주행 단위", h.group), (f"주행 안 {BLOCK}장 묶음", h.block)):
        print(f"  {name:12s}", repeated(h, grp))

    print("\n② 카메라 하나 빼고 학습 → 뺀 카메라로 시험")
    allc = pd.concat([g, k, h[["group", "camera", "label"] + FEATURES]], ignore_index=True)
    fit = lambda tr: lightgbm_model().fit(tr[FEATURES], tr.label)
    base = fit(g)
    rows = []
    for cam in ("블랙박스", "도로 광각", "AI Hub", "GLARE"):
        te = allc[allc.camera.eq(cam)]
        y = te.label.to_numpy()
        if cam != "GLARE":
            rows.append({"시험": cam, "학습": "GLARE 만", **scores(y, base.predict(te[FEATURES]))})
        rows.append({"시험": cam, "학습": "나머지 카메라 전부", **scores(y, fit(allc[~allc.camera.eq(cam)]).predict(te[FEATURES]))})
    r = pd.DataFrame(rows)
    for v in ("macro F1", "심함 재현율", "눈부심 여부 정확도"):
        print(f"\n  {v}")
        print(r.pivot_table(index="시험", columns="학습", values=v, sort=False).round(2).to_string())

    m = fit(allc)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES)
    print("\n네 카메라 전부로 학습한 모델의 중요 특징:", (imp / imp.sum()).sort_values(ascending=False).head(5).round(2).to_dict())


if __name__ == "__main__":
    main()
