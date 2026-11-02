"""모델 ② 재학습 실험: GLARE(미국 130장) 에 한국 라벨 사진을 넣으면 한국 사진에서 나아지는가.

같은 촬영(같은 카메라 · 같은 날, 없으면 묶음) 사진이 학습과 시험에 섞이지 않게 촬영 단위 GroupKFold(5).
2019-02-24 블랙박스 여행은 한 덩어리 → 한 폴드에 통째로 들어감.
비교하는 학습 데이터 (모델 · 특징 12개 · 설정은 배포 모델과 같음, pilot_eval.lightgbm_model):
    GLARE        GLARE 130장만 (지금 배포 방식)
    한국         한국 학습 폴드만
    GLARE+한국   둘 다

    python notebooks/experiments/glare_video_source/retrain_korea.py --glare ../../data/glare/pilot_glare
결과: data/glare_mapillary/retrain_oof.csv (사진별 촬영 밖 예측)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupKFold

from nunchi_ml.glare.features import FEATURES
from nunchi_ml.glare.pilot_eval import load, lightgbm_model

import find_sun_frames as F

OUT = F.OUT


def korea():
    c = pd.read_csv(OUT / "candidates.csv")
    day = pd.to_datetime(c.captured_at_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    trip = pd.Series(np.where(c.make.fillna("none").eq("none"), c.sequence,
                              c.make.astype(str) + " " + c.model.astype(str) + " " + day), index=c.id)
    drive = pd.read_csv(OUT / "labels_drive.csv").assign(src="drive")
    dash = pd.read_csv(OUT / "labels_dash.csv").assign(src="dash")
    d = pd.concat([drive, dash], ignore_index=True)[["id", "src", "label"] + FEATURES]
    d["group"] = "kr:" + d.id.map(trip)
    d["camera"] = np.where(d.group.str.contains("iPhone"), "블랙박스", "도로 광각")
    return d


def scores(y, p):
    return {"사진": len(y), "macro F1": f1_score(y, p, average="macro", labels=[0, 1, 2]),
            "심함 재현율": recall_score(y, p, labels=[2], average="macro", zero_division=0),
            "눈부심 여부 정확도": np.mean((y >= 1) == (p >= 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glare", required=True, help="glare_features.csv · pilot_labels.csv 가 있는 폴더")
    a = ap.parse_args()
    g = load(Path(a.glare) / "glare_features.csv", Path(a.glare) / "pilot_labels.csv")
    g = g.assign(group="us:" + g.vid, camera="GLARE")[["group", "camera", "label"] + FEATURES]
    k = korea()
    print(f"GLARE {len(g)}장 · 한국 {len(k)}장 (촬영 {k.group.nunique()}개, 블랙박스 여행 {int((k.camera == '블랙박스').sum())}장)"
          f" · 한국 라벨 {k.label.value_counts().sort_index().to_dict()}")

    fit = lambda tr: lightgbm_model().fit(tr[FEATURES], tr.label)
    oof = {name: np.zeros(len(k), int) for name in ("GLARE", "한국", "GLARE+한국")}
    base = fit(g)
    oof["GLARE"] = base.predict(k[FEATURES])
    for tr, te in GroupKFold(n_splits=5).split(k, k.label, k.group):
        oof["한국"][te] = fit(k.iloc[tr]).predict(k.iloc[te][FEATURES])
        oof["GLARE+한국"][te] = fit(pd.concat([g, k.iloc[tr]])).predict(k.iloc[te][FEATURES])

    rows = []
    for name, p in oof.items():
        for cam in ("블랙박스", "도로 광각", "한국 전체"):
            m = k.camera.eq(cam) if cam != "한국 전체" else np.ones(len(k), bool)
            rows.append({"학습": name, "시험": cam, **scores(k.label.to_numpy()[m], p[m])})
    # GLARE 쪽이 나빠지지 않는지: GLARE 촬영 밖 예측 (한국은 전부 학습에)
    pg = {"GLARE": np.zeros(len(g), int), "GLARE+한국": np.zeros(len(g), int)}
    for tr, te in GroupKFold(n_splits=5).split(g, g.label, g.group):
        pg["GLARE"][te] = fit(g.iloc[tr]).predict(g.iloc[te][FEATURES])
        pg["GLARE+한국"][te] = fit(pd.concat([g.iloc[tr], k])).predict(g.iloc[te][FEATURES])
    for name, p in pg.items():
        rows.append({"학습": name, "시험": "GLARE", **scores(g.label.to_numpy(), p)})
    r = pd.DataFrame(rows)
    print(r.pivot_table(index="시험", columns="학습", values="macro F1", sort=False).round(2).to_string())
    print("\n심함 재현율"); print(r.pivot_table(index="시험", columns="학습", values="심함 재현율", sort=False).round(2).to_string())
    print("\n눈부심 여부(0 vs 1+) 정확도"); print(r.pivot_table(index="시험", columns="학습", values="눈부심 여부 정확도", sort=False).round(2).to_string())
    k.assign(**{f"pred_{n}": p for n, p in oof.items()}).to_csv(OUT / "retrain_oof.csv", index=False, encoding="utf-8-sig")

    m = fit(pd.concat([g, k]))
    imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES)
    print("\nGLARE+한국 전체로 학습한 모델의 중요 특징:", (imp / imp.sum()).sort_values(ascending=False).head(5).round(2).to_dict())
    probes(g, k, base)


def probes(g, k, base):
    """원인 가리기: ① 화면 비율 ② 해 위치 특징 ③ 같은 카메라 안에서는 맞히는가."""
    from sklearn.model_selection import StratifiedKFold
    import cv2
    from nunchi_ml.glare.features import extract_array
    meta = pd.concat([pd.read_csv(OUT / n)[["id", "file", "sun_alt", "gap", "heading_suspect"]]
                      for n in ("labels_drive.csv", "labels_dash.csv")]).drop_duplicates("id")
    k = k.merge(meta, on="id")
    k["gap_c"] = np.where(k.heading_suspect, np.nan, k.gap)          # 방향 기록이 틀린 사진은 Δ 모름
    f1 = lambda y, p, m: round(scores(y[m], p[m])["macro F1"], 2)
    cams = ("블랙박스", "도로 광각")

    def crop169(f):                                                    # GLARE 화면 비율(937×540)로 가운데 자르기
        im = cv2.imdecode(np.fromfile(str(OUT / f), np.uint8), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        nh = int(w * 540 / 937)
        y0 = max(0, (h - nh) // 2)
        return extract_array(im[y0:y0 + nh])
    kc = k.copy()
    kc[FEATURES] = pd.DataFrame([crop169(f) for f in k.file])[FEATURES].to_numpy()
    print("\n① 화면 비율 — GLARE 모델로 채점 (macro F1)")
    for name, kk in (("4:3 그대로 늘림", k), ("16:9 로 자름", kc)):
        p = base.predict(kk[FEATURES])
        print(f"  {name}: " + " · ".join(f"{c} {f1(kk.label.to_numpy(), p, kk.camera.eq(c).to_numpy())}" for c in cams))

    sets = (("화면 12개", FEATURES), ("해 위치 2개(높이 · Δ)", ["sun_alt", "gap_c"]), ("화면 12 + 해 위치", FEATURES + ["sun_alt", "gap_c"]))
    print("\n② 해 위치 특징 — 한국만 학습, 촬영 단위 CV (macro F1)")
    for name, cols in sets:
        p = np.zeros(len(k), int)
        for tr, te in GroupKFold(n_splits=5).split(k, k.label, k.group):
            p[te] = lightgbm_model().fit(k.iloc[tr][cols], k.iloc[tr].label).predict(k.iloc[te][cols])
        print(f"  {name}: " + " · ".join(f"{c} {f1(k.label.to_numpy(), p, k.camera.eq(c).to_numpy())}" for c in cams))

    b = k[k.camera.eq("블랙박스")].reset_index(drop=True)
    print("\n③ 블랙박스 여행 안에서만 사진 단위 CV (같은 날이라 낙관적 — 특징끼리 비교용)")
    for name, cols in sets:
        p = np.zeros(len(b), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(b, b.label):
            p[te] = lightgbm_model().fit(b.iloc[tr][cols], b.iloc[tr].label).predict(b.iloc[te][cols])
        s = scores(b.label.to_numpy(), p)
        print(f"  {name}: macro F1 {s['macro F1']:.2f} · 심함 재현율 {s['심함 재현율']:.2f}")


if __name__ == "__main__":
    main()
