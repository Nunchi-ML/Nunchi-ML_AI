"""카메라 화각을 반영한 해 위치 특징 실험: '해가 이 카메라 화면 안 어디쯤 있나' 를 넣으면 처음 보는 카메라에서도 나아지는가.

Mapillary 가 사진마다 계산해 주는 초점거리(camera_parameters[0], 긴 변 대비) → 가로 · 세로 화각.
카메라가 수평 · 진행 방향을 본다고 보고, 해를 화면에 투영한 위치:
    u = tan(Δ부호) / tan(가로 화각/2)      −1 ~ 1 이면 가로로 화면 안 (0 = 가운데)
    v = tan(해 높이) / tan(세로 화각/2)     0 ~ 1 이면 세로로 화면 안 (0 = 지평선)
    in_fov = |u| ≤ 1 · 0 < v ≤ 1 · Δ < 90°,  r = √(u² + v²)  (화면 가운데에서 해까지)
비교 (한국 284장만 — GLARE 는 촬영 정보 없음, 촬영 단위 GroupKFold(5)):
    화면 12 · 해 위치(높이 · Δ) · 화각 반영 해 위치(u · v · r · in_fov) · 화면 12 + 화각 반영

    python notebooks/experiments/glare_video_source/fov_probe.py
결과: data/glare_mapillary/camera_params.csv (사진별 초점거리 캐시)
"""
import time

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import GroupKFold

from nunchi_ml.glare.features import FEATURES
from nunchi_ml.glare.pilot_eval import lightgbm_model

import find_sun_frames as F
import retrain_korea as R

OUT = F.OUT


def camera_params(ids, out=None):
    cache = (out or OUT) / "camera_params.csv"
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["id", "focal", "width", "height", "camera_type"])
    tok, rows = F.token(), []
    for n, i in enumerate([i for i in ids if i not in set(have.id)], 1):
        r = requests.get(f"https://graph.mapillary.com/{i}", timeout=60, params={
            "access_token": tok, "fields": "camera_parameters,width,height,camera_type"}).json()
        cp = r.get("camera_parameters") or [np.nan]
        rows.append({"id": i, "focal": cp[0], "width": r.get("width"), "height": r.get("height"), "camera_type": r.get("camera_type")})
        if n % 50 == 0:
            print(f"  화각 {n}장", flush=True)
        time.sleep(0.05)
    have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    have.to_csv(cache, index=False)
    return have


def fov_features(d):
    d = d.astype({"focal": float, "width": float, "height": float})
    long = np.maximum(d.width, d.height)
    fpx = d.focal * long
    hf, vf = np.arctan(d.width / 2 / fpx), np.arctan(d.height / 2 / fpx)     # 반화각 (라디안)
    side = ((d.sun_az - d.heading + 540) % 360) - 180
    s = np.radians(side.clip(-89, 89))
    u = np.tan(s) / np.tan(hf)
    v = np.tan(np.radians(d.sun_alt.clip(-89, 89))) / np.tan(vf) / np.cos(s)
    bad = d.heading_suspect.astype(bool)                                     # 방향 기록이 틀린 사진은 모름
    out = pd.DataFrame({"hfov": np.degrees(2 * hf), "u": u, "v": v, "r": np.hypot(u, v),
                        "in_fov": ((u.abs() <= 1) & (v > 0) & (v <= 1) & (side.abs() < 90)).astype(float)})
    out.loc[bad, ["u", "v", "r", "in_fov"]] = np.nan
    return out


def main():
    k = R.korea()
    meta = pd.concat([pd.read_csv(OUT / n)[["id", "sun_alt", "sun_az", "heading", "gap", "heading_suspect"]]
                      for n in ("labels_drive.csv", "labels_dash.csv")]).drop_duplicates("id")
    k = k.merge(meta, on="id")
    cp = camera_params(k.id.tolist())
    k = k.merge(cp[["id", "focal", "width", "height"]], on="id")
    k = pd.concat([k.reset_index(drop=True), fov_features(k).reset_index(drop=True)], axis=1)
    k["gap_c"] = np.where(k.heading_suspect, np.nan, k.gap)
    print("가로 화각(°) 카메라별:", k.groupby("camera").hfov.describe()[["min", "50%", "max"]].round(0).to_dict("index"))
    print("해가 화면 안(in_fov) 사진별 눈부심 비율:")
    print(k.groupby(["camera", "in_fov"], dropna=False).agg(사진=("label", "size"), 눈부심=("label", lambda x: (x >= 1).mean()),
                                                         심함=("label", lambda x: (x == 2).mean())).round(2).to_string())

    sets = {"화면 12개": FEATURES, "해 위치 (높이 · Δ)": ["sun_alt", "gap_c"],
            "화각 반영 해 위치": ["u", "v", "r", "in_fov", "sun_alt"],
            "화면 12 + 화각 반영": FEATURES + ["u", "v", "r", "in_fov", "sun_alt"]}
    y, rows = k.label.to_numpy(), []
    for name, cols in sets.items():
        p = np.zeros(len(k), int)
        for tr, te in GroupKFold(n_splits=5).split(k, y, k.group):
            p[te] = lightgbm_model().fit(k.iloc[tr][cols], y[tr]).predict(k.iloc[te][cols])
        for cam in ("블랙박스", "도로 광각"):
            m = k.camera.eq(cam).to_numpy()
            s = R.scores(y[m], p[m])
            rows.append({"특징": name, "시험": cam, "macro F1": s["macro F1"], "심함 재현율": s["심함 재현율"],
                         "눈부심 여부": s["눈부심 여부 정확도"]})
    r = pd.DataFrame(rows)
    for col in ("macro F1", "심함 재현율", "눈부심 여부"):
        print(f"\n{col}"); print(r.pivot_table(index="특징", columns="시험", values=col, sort=False).round(2).to_string())


if __name__ == "__main__":
    main()
