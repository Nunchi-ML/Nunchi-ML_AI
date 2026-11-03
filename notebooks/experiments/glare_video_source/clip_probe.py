"""기법 바꾸기 실험: 밝기 숫자 12개 대신 사전학습 이미지 모델(CLIP) 특징으로 눈부심 3단계를 맞히면 처음 보는 카메라에서도 나아지는가.

CLIP(openai/clip-vit-base-patch32) 은 수많은 사진 · 설명으로 미리 학습돼 '역광' · '해가 보임' 같은 장면 내용을 잡음.
비교 (retrain_korea.py 와 같은 사진 · 같은 촬영 단위 GroupKFold(6)):
    숫자 12개 + LightGBM      지금 모델 ② 방식
    CLIP 특징 + 로지스틱       사진 → 512차원 → 선형 분류
    CLIP 글 설명 비교 (학습 없음)  "맑은 도로" · "눈부신 햇빛" · "해가 정면이라 앞이 안 보임" 과 가장 비슷한 쪽

PyTorch 가 커서 별도 venv:
    python -m venv .venv-clip && .venv-clip/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv-clip/Scripts/pip install transformers -e .
    .venv-clip/Scripts/python notebooks/experiments/glare_video_source/clip_probe.py --glare ../../data/glare/pilot_glare
결과: data/glare_mapillary/clip_embeddings.npz (사진별 특징 캐시)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from nunchi_ml.glare.features import FEATURES
from nunchi_ml.glare.pilot_eval import load, lightgbm_model

import retrain_korea as R

OUT = R.OUT
MODEL = "openai/clip-vit-base-patch32"
PROMPTS = ["a clear photo of a road taken from a car, no sun glare",
           "a road photo from a car with bright sun glare, but the road is still visible",
           "a road photo from a car blinded by the sun straight ahead, the road is washed out"]


def vec(out):
    """transformers 4 는 텐서, 5 는 출력 객체(pooler_output 이 투영된 특징)를 돌려줌"""
    return out if hasattr(out, "norm") else out.pooler_output


def embed(paths, cache):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    have = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    keys = [str(p) for p in paths]
    todo = [p for p in keys if p not in have]
    model = CLIPModel.from_pretrained(MODEL).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    with torch.no_grad():
        for i in range(0, len(todo), 32):
            batch = todo[i:i + 32]
            x = proc(images=[Image.open(p).convert("RGB") for p in batch], return_tensors="pt")
            e = vec(model.get_image_features(**x))
            e = (e / e.norm(dim=-1, keepdim=True)).numpy()
            have.update(zip(batch, e))
            print(f"  CLIP {min(i + 32, len(todo))}/{len(todo)}", flush=True)
        t = proc(text=PROMPTS, return_tensors="pt", padding=True)
        txt = vec(model.get_text_features(**t))
        txt = (txt / txt.norm(dim=-1, keepdim=True)).numpy()
    np.savez(cache, **have)
    return np.stack([have[p] for p in keys]), txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glare", required=True, help="glare_features.csv · pilot_labels.csv 가 있는 폴더 (사진은 그 위 폴더 기준 path 열)")
    a = ap.parse_args()
    gdir = Path(a.glare)
    g = load(gdir / "glare_features.csv", gdir / "pilot_labels.csv")
    g = g.assign(group="us:" + g.vid, camera="GLARE", img=[str((gdir.parent / p).resolve()) for p in g.path])
    k = R.korea()
    files = pd.concat([pd.read_csv(OUT / n)[["id", "file"]] for n in ("labels_drive.csv", "labels_dash.csv")]).drop_duplicates("id")
    k = k.merge(files, on="id").assign(img=lambda d: [str(OUT / f) for f in d.file])
    d = pd.concat([g[["group", "camera", "label", "img"] + FEATURES], k[["group", "camera", "label", "img"] + FEATURES]], ignore_index=True)
    print(f"사진 {len(d)}장 (GLARE {len(g)} · 한국 {len(k)})")

    E, T = embed(d.img, OUT / "clip_embeddings.npz")
    y = d.label.to_numpy()
    preds = {"숫자 12개 + LightGBM": np.zeros(len(d), int), "CLIP + 로지스틱": np.zeros(len(d), int)}
    for tr, te in GroupKFold(n_splits=6).split(d, y, d.group):
        preds["숫자 12개 + LightGBM"][te] = lightgbm_model().fit(d.iloc[tr][FEATURES], y[tr]).predict(d.iloc[te][FEATURES])
        clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(E[tr], y[tr])
        preds["CLIP + 로지스틱"][te] = clf.predict(E[te])
    preds["CLIP 글 설명 비교 (학습 없음)"] = (E @ T.T).argmax(1)

    rows = []
    for name, p in preds.items():
        for cam in ("블랙박스", "도로 광각", "GLARE"):
            m = d.camera.eq(cam).to_numpy()
            s = R.scores(y[m], p[m])
            rows.append({"방법": name, "시험": cam, "macro F1": s["macro F1"], "심함 재현율": s["심함 재현율"]})
    r = pd.DataFrame(rows)
    print("\nmacro F1"); print(r.pivot_table(index="방법", columns="시험", values="macro F1", sort=False).round(2).to_string())
    print("\n심함 재현율"); print(r.pivot_table(index="방법", columns="시험", values="심함 재현율", sort=False).round(2).to_string())


if __name__ == "__main__":
    main()
