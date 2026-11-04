"""상대 특징 12개를 줄인다 — 선택은 세 카메라(도로 광각 · 블랙박스 · 서울 폰)로만, AI Hub 는 마지막에 시험용으로만 본다.

기준 점수 = (카메라 하나 빼고 시험 눈부심 여부 평균 + 촬영 단위 CV(5 × 10) 눈부심 여부) / 2   (두 단계, 상대 특징 부분집합)
  1) 하나씩 빼 보기 → 빠졌을 때 점수가 얼마나 떨어지나
  2) 후진 제거: 점수가 가장 덜 떨어지는 특징부터 계속 빼서 12 → 3 개
  3) 크기별 최선 부분집합을 AI Hub(외부 시험) 로 채점 — 선택에 쓰지 않은 카메라

    python notebooks/experiments/sun_visible_probe/ablation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import sun_visible_probe as S
from rel_features import REL
import retrain_korea as R

REPEATS = 10


def acc(y, p):
    return float(np.mean((y >= 1) == (p >= 1)))


def score(d3, cols):
    y = d3.label.to_numpy()
    loco = []
    for c in S.THREE:
        te = d3.camera.eq(c).to_numpy()
        loco.append(acc(y[te], S.two_stage(d3[~te], d3[te], cols)))
    cv = []
    for r in range(REPEATS):
        pred = np.zeros(len(d3), int)
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(d3, y, d3.group):
            pred[te] = S.two_stage(d3.iloc[tr], d3.iloc[te], cols)
        cv.append(acc(y, pred))
    return (np.mean(loco) + np.mean(cv)) / 2, np.mean(loco), min(loco), np.mean(cv)


def main():
    d = S.data()
    d3 = d[d.camera.isin(S.THREE)].reset_index(drop=True)
    ah = d[d.camera.eq("AI Hub")].reset_index(drop=True)
    ya = ah.label.to_numpy()

    full = score(d3, REL)
    print(f"전체 12개: 기준 {full[0]:.3f} (카메라 빼고 평균 {full[1]:.2f} · 최저 {full[2]:.2f} · CV {full[3]:.2f})\n")

    print("1) 하나씩 빼 보기 (기준 점수 변화 · 음수 = 빼니 나빠짐 = 중요)")
    rows = []
    for f in REL:
        s = score(d3, [x for x in REL if x != f])
        rows.append((f, s[0] - full[0]))
    for f, dlt in sorted(rows, key=lambda x: x[1]):
        print(f"  {f:10s} {dlt:+.3f}")

    print("\n2) 후진 제거 (세 카메라 점수만 사용)")
    cur, path = list(REL), {}
    path[len(cur)] = list(cur)
    while len(cur) > 3:
        best = max(cur, key=lambda f: score(d3, [x for x in cur if x != f])[0])
        cur.remove(best)
        path[len(cur)] = list(cur)
        s = score(d3, cur)
        print(f"  {len(cur):2d}개 (뺀 것 {best:10s}) 기준 {s[0]:.3f} · 평균 {s[1]:.2f} · 최저 {s[2]:.2f} · CV {s[3]:.2f}")

    print("\n3) 크기별 부분집합을 AI Hub(선택에 안 쓴 카메라)로 채점: 여부 · 헛알림 · 재현율")
    for k in sorted(path, reverse=True):
        p = S.two_stage(d3, ah, path[k])
        fa, rc = S.extra(ya, p)
        print(f"  {k:2d}개 여부 {acc(ya, p):.2f} · 헛알림 {fa:.2f} · 재현율 {rc:.2f} | {', '.join(path[k]) if k <= 6 else ''}")


if __name__ == "__main__":
    main()
