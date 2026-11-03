"""배포 중인 모델 ②(GLARE 미국 사진으로 학습)를 한국 도로 라벨 사진에 그대로 채점.

모델 번들은 버전을 엄격히 확인하므로 배포와 같은 환경(BE venv)으로 돌린다:
    ../BE/.venv/Scripts/python notebooks/experiments/glare_video_source/evaluate_korea.py --model ../BE/models/model2-pilot-v1
채점 세트 (make_drive_set.py · analyze_dashcam.py 결과):
    drive   운전 시점 124장 (성진 라벨) · drive_agree 그중 두 사람 합의 94장 (labels_merged.csv)
    dash    블랙박스 160장 (성진 라벨, 방향 의심도 포함 — 화면만 보는 모델이라)
결과: data/glare_mapillary/korea_predictions.csv (사진별 예측 · 점수)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from nunchi_ml.glare.inference import GlarePredictor

OUT = Path(__file__).resolve().parents[3] / "data" / "glare_mapillary"


def report(name, y, p):
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    print(f"\n[{name}] {len(y)}장 · macro F1 {f1_score(y, p, average='macro', labels=[0, 1, 2]):.2f} · "
          f"심함 재현율 {recall_score(y, p, labels=[2], average='macro', zero_division=0):.2f} · "
          f"눈부심 여부(0 vs 1+) 정확도 {np.mean((y >= 1) == (p >= 1)):.2f}")
    print("          예측 0   1   2")
    for i in range(3):
        print(f"  정답 {i}  " + "".join(f"{cm[i, j]:5d}" for j in range(3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    model = GlarePredictor(a.model)
    print("모델", model.meta["model_version"], "· 학습:", model.meta["source"]["dataset"], model.meta["labels"]["count"], "장")

    drive = pd.read_csv(OUT / "labels_drive.csv")[["id", "file", "sequence", "label"]].assign(set="drive")
    dash = pd.read_csv(OUT / "labels_dash.csv")[["id", "file", "sequence", "label"]].assign(set="dash")
    d = pd.concat([drive, dash], ignore_index=True)
    preds = []
    for f in d.file:
        r = model.predict(cv2.imdecode(np.fromfile(str(OUT / f), np.uint8), cv2.IMREAD_COLOR))
        preds.append({"pred": r["level"], **{f"p_{k}": v for k, v in r["scores"].items()}})
    d = pd.concat([d, pd.DataFrame(preds)], axis=1)
    d.to_csv(OUT / "korea_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n예측 분포", d.groupby("set").pred.value_counts().unstack(fill_value=0).to_dict("index"))
    for name, g in d.groupby("set", sort=False):
        report(name, g.label.to_numpy(), g.pred.to_numpy())
    agree = pd.read_csv(OUT / "labels_merged.csv", dtype={"label": str})
    agree = agree[agree.label.isin(["0", "1", "2"])][["id", "label"]].rename(columns={"label": "agree"})
    g = d[d.set == "drive"].merge(agree, on="id")
    report("drive_agree (두 사람 합의)", g.agree.astype(int).to_numpy(), g.pred.to_numpy())
    print("\n참고: GLARE 안에서의 교차검증 macro F1", round(model.meta["validation"]["raw"]["macro_f1"], 2),
          "· 심함 재현율", round(model.meta["validation"]["raw"]["severe_recall"], 2))


if __name__ == "__main__":
    main()
