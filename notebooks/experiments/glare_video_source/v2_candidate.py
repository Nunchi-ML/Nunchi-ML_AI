"""모델 ② v2 후보: 화면 특징 12개 + 화각 반영 해 위치(u · v · r · in_fov · 해 높이) 를 한 모델로.

GLARE(미국 130장) 는 촬영 정보가 없어 해 위치 칸이 빈칸(NaN) — LightGBM 은 빈칸을 그대로 다룸.
앱에서도 GPS · 나침반을 못 받으면 빈칸으로 넣으면 되므로 같은 모델 하나로 두 경우를 다 처리.
비교 (GLARE + 한국 414장, 촬영 단위 GroupKFold(6)):
    v1      화면 12개, GLARE+한국 (지금 방식에 한국만 더함)
    v2      화면 12 + 해 위치, GLARE+한국
    v2-빈칸  v2 모델에 해 위치를 일부러 비워서 넣었을 때 (앱이 위치 · 방향을 못 받는 경우)

    python notebooks/experiments/glare_video_source/v2_candidate.py --glare ../../data/glare/pilot_glare
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from nunchi_ml.glare.features import FEATURES
from nunchi_ml.glare.pilot_eval import load, lightgbm_model

import find_sun_frames as F
import fov_probe as P
import retrain_korea as R

OUT = F.OUT
SUN = ["u", "v", "r", "in_fov", "sun_alt"]


def korea_with_sun():
    k = R.korea()
    meta = pd.concat([pd.read_csv(OUT / n)[["id", "sun_alt", "sun_az", "heading", "gap", "heading_suspect"]]
                      for n in ("labels_drive.csv", "labels_dash.csv")]).drop_duplicates("id")
    k = k.merge(meta, on="id").merge(P.camera_params(k.id.tolist())[["id", "focal", "width", "height"]], on="id")
    k = pd.concat([k.reset_index(drop=True), P.fov_features(k).reset_index(drop=True)], axis=1)
    k.loc[k.heading_suspect.astype(bool), "sun_alt"] = np.nan        # 방향을 모르면 해 위치 전체를 빈칸
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glare", required=True)
    a = ap.parse_args()
    g = load(Path(a.glare) / "glare_features.csv", Path(a.glare) / "pilot_labels.csv")
    g = g.assign(group="us:" + g.vid, camera="GLARE", **{c: np.nan for c in SUN})
    k = korea_with_sun()
    cols = ["group", "camera", "label"] + FEATURES + SUN
    d = pd.concat([g[cols], k[cols]], ignore_index=True)
    y = d.label.to_numpy()
    print(f"사진 {len(d)}장 (GLARE {len(g)} · 한국 {len(k)}, 해 위치 있는 사진 {int(d.u.notna().sum())})")

    preds = {n: np.zeros(len(d), int) for n in ("v1 화면 12", "v2 화면 12 + 해 위치", "v2 · 해 위치 빈칸으로 넣음")}
    for tr, te in GroupKFold(n_splits=6).split(d, y, d.group):
        m1 = lightgbm_model().fit(d.iloc[tr][FEATURES], y[tr])
        m2 = lightgbm_model().fit(d.iloc[tr][FEATURES + SUN], y[tr])
        preds["v1 화면 12"][te] = m1.predict(d.iloc[te][FEATURES])
        preds["v2 화면 12 + 해 위치"][te] = m2.predict(d.iloc[te][FEATURES + SUN])
        blank = d.iloc[te][FEATURES + SUN].copy(); blank[SUN] = np.nan
        preds["v2 · 해 위치 빈칸으로 넣음"][te] = m2.predict(blank)

    rows = []
    for name, p in preds.items():
        for cam in ("블랙박스", "도로 광각", "GLARE"):
            m = d.camera.eq(cam).to_numpy()
            s = R.scores(y[m], p[m])
            rows.append({"모델": name, "시험": cam, **{k_: s[k_] for k_ in ("macro F1", "심함 재현율", "눈부심 여부 정확도")}})
    r = pd.DataFrame(rows)
    for col in ("macro F1", "심함 재현율", "눈부심 여부 정확도"):
        print(f"\n{col}"); print(r.pivot_table(index="모델", columns="시험", values=col, sort=False).round(2).to_string())

    m = lightgbm_model().fit(d[FEATURES + SUN], y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES + SUN)
    print("\nv2 전체 학습 중요 특징:", (imp / imp.sum()).sort_values(ascending=False).head(6).round(2).to_dict())


if __name__ == "__main__":
    main()
