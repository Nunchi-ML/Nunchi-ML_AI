"""배포 중인 모델 ② 를 AI Hub 역광 라벨 세트(3인 다수결)에 그대로 채점 — 처음 보는 카메라 하나 더.

배포와 같은 환경(BE venv)으로:
    ../BE/.venv/Scripts/python notebooks/experiments/glare_video_source/evaluate_aihub.py --model ../BE/models/model2-pilot-v1
입력: data/aihub_special/labels_merged.csv (compare_aihub_labels.py 결과)
결과: data/aihub_special/aihub_predictions.csv
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate_korea import report
from nunchi_ml.glare.inference import GlarePredictor

OUT = Path(__file__).resolve().parents[3] / "data" / "aihub_special"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    model = GlarePredictor(ap.parse_args().model)
    print("모델", model.meta["model_version"])

    d = pd.read_csv(OUT / "labels_merged.csv")
    d = d[d.majority.notna()].reset_index(drop=True)
    preds = []
    for f in d.file:
        r = model.predict(cv2.imdecode(np.fromfile(str(OUT / f), np.uint8), cv2.IMREAD_COLOR))
        preds.append({"pred": r["level"], **{f"p_{k}": v for k, v in r["scores"].items()}})
    d = pd.concat([d, pd.DataFrame(preds)], axis=1)
    d["majority"] = d.majority.astype(int)
    d.to_csv(OUT / "aihub_predictions.csv", index=False, encoding="utf-8-sig")

    print("예측 분포", d.pred.value_counts().sort_index().to_dict())
    report("AI Hub 전체 (다수결)", d.majority.to_numpy(), d.pred.to_numpy())
    u = d[d.unanimous]
    report("AI Hub 만장일치만", u.majority.to_numpy(), u.pred.to_numpy())
    for g, x in d.groupby("group"):
        report(g, x.majority.to_numpy(), x.pred.to_numpy())


if __name__ == "__main__":
    main()
