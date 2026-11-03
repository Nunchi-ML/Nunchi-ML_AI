"""서울 전역 Mapillary → G (해 위치 + 날씨) 학습용 라벨 세트 (운전 시점만, 약 200장).

서울 Mapillary 정면 사진은 대부분 인도 · 산책로 보행 촬영이라, 묶음 이동 속도(GPS · 시각 중앙값)가
MIN_KMH 이상인 묶음(차량)만 쓰고, 그중 눈으로 운전 시점(차 앞유리 · 버스 앞)으로 본 묶음만 남긴다
(sequence_view_seoul.csv — 첫 실행 결과 53묶음 판정: 기차 창밖 7 · 옆 창 13 · 보행 등 11 · 운전 22). 정면만 모으면 '언제 눈부심이 아닌가' 를 못 배우므로 해 위치를 고루 섞는다.
시각 확인(clock_ok) · 방향 믿을 수 있는(heading_suspect 아님) 사진만, 묶음마다 3초 이상 간격 · 장수 상한.

    group        해 높이     Δ          목표   묶음당 최대
    front        0~25°      ≤ 25°       60   15     정면 저각도 (운전 시점은 6묶음뿐 → 거의 전부)
    near         0~25°      25~45°      80    6     광각이면 화면 안
    side_back    0~25°      ≥ 60°       60    2     해는 낮지만 옆 · 뒤 (0 예시)
    high_front   25~50°     ≤ 45°       20    2     해가 높음 (해 높이 효과)

    GLARE_SET=glare_mapillary_seoul python notebooks/experiments/glare_video_source/make_seoul_label_set.py
먼저 find_sun_frames.py (data/roads/seoul_grid.json) → check_clock.py → fix_metadata.py
결과 (data/glare_mapillary_seoul/, git 제외):
    images/<묶음>/<id>.jpg · label_set.csv (분석용)
    seoul_pack/ · seoul_pack.zip  라벨러용: 사진 + 무작위 순서 label_sheet_blind.csv + label_tool.html + attribution.csv
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import find_sun_frames as F
from download_label_set import fetch

OUT = F.OUT
HERE = Path(__file__).parent
GAP_S, SEED = 3, 0
MIN_KMH = 20                      # 걷기 · 자전거 묶음 빼기 (정면 묶음 속도: 8 이하 35 · 8~20 12 · 30 이상 6, 사이는 0)
GROUPS = [  # 이름, 해 높이 범위, Δ 범위, 목표 장수, 묶음당 최대
    ("front", (0, 25), (0, 25), 60, 15),
    ("near", (0, 25), (25, 45), 80, 6),
    ("side_back", (0, 25), (60, 180), 60, 2),
    ("high_front", (25, 50), (0, 45), 20, 2),
]


def sequence_speed(c):
    """묶음별 이동 속도 중앙값 (km/h). 30초 넘게 끊긴 구간 · 사진 3장 미만 묶음은 nan."""
    def one(g):
        if len(g) < 3:
            return np.nan
        lat = np.radians(g.lat.to_numpy())
        x, y = 6371000 * np.radians(g.lon.to_numpy()) * np.cos(lat.mean()), 6371000 * lat
        d, t = np.hypot(np.diff(x), np.diff(y)), np.diff(g.captured_at_ms.to_numpy()) / 1000
        m = (t > 0) & (t < 30)
        return np.nan if m.sum() < 2 else float(np.median(d[m] / t[m]) * 3.6)
    return c.sort_values("captured_at_ms").groupby("sequence")[["lat", "lon", "captured_at_ms"]].apply(one).rename("kmh")


def spaced(t, cap):
    """묶음마다 GAP_S 초 이상 떨어진 사진만, 묶음당 cap 장 (묶음 안에서 고르게)."""
    keep = []
    for _, g in t.sort_values("captured_at_ms").groupby("sequence"):
        last, idx = None, []
        for r in g.itertuples():
            if last is None or r.captured_at_ms - last >= GAP_S * 1000:
                idx.append(r.Index); last = r.captured_at_ms
        step = max(1, len(idx) / cap)
        keep += [idx[int(i * step)] for i in range(min(cap, len(idx)))]
    return t.loc[keep]


def pick(c):
    c = c.merge(sequence_speed(c), left_on="sequence", right_index=True)
    ok = c[c.clock_ok & ~c.heading_suspect.astype(bool) & (c.kmh >= MIN_KMH)]
    view = HERE / "sequence_view_seoul.csv"           # 눈으로 본 묶음 시점 (속도만으론 기차 창밖 · 옆 창이 남음)
    if view.exists():
        v = pd.read_csv(view)
        ok = ok[ok.sequence.isin(v.sequence[v.view == "drive"])]
    out = []
    for name, (a0, a1), (g0, g1), n, cap in GROUPS:
        t = spaced(ok[(ok.sun_alt > a0) & (ok.sun_alt <= a1) & (ok.gap > g0 - (g0 == 0)) & (ok.gap <= g1)], cap)
        t = t.sample(min(n, len(t)), random_state=SEED)
        print(f"  {name:10s} {len(t):3d}장 · 묶음 {t.sequence.nunique()} · 촬영일 {t.captured_at_kst.str[:10].nunique()}")
        out.append(t.assign(group=name))
    return pd.concat(out)


def main():
    c = pd.read_csv(OUT / "candidates.csv")
    sel = pick(c)
    sel = sel.merge(fetch(sel, F.token()), on="id")
    sel = sel[sel.file.notna()]
    cols = ["id", "file", "group", "sequence", "kmh", "captured_at_kst", "lat", "lon", "heading", "track",
            "sun_alt", "sun_az", "gap", "front_low", "make", "model", "creator"]
    sel[cols].to_csv(OUT / "label_set.csv", index=False, encoding="utf-8-sig")

    p = OUT / "seoul_pack"
    shutil.rmtree(p, ignore_errors=True)
    for f in sel.file:
        (p / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT / f, p / f)
    blind = sel[["id", "file", "sequence"]].sample(frac=1, random_state=SEED).reset_index(drop=True)
    blind.insert(0, "no", range(1, len(blind) + 1))
    blind.assign(label="", confidence="", note="").to_csv(p / "label_sheet_blind.csv", index=False, encoding="utf-8-sig")
    sel.assign(url="https://www.mapillary.com/app/?pKey=" + sel.id.astype(str))[["id", "creator", "captured_at_kst", "url"]] \
        .to_csv(p / "attribution.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(HERE / "label_tool.html", p / "label_tool.html")
    z = shutil.make_archive(str(OUT / "seoul_pack"), "zip", p)
    print(f"받은 사진 {len(sel)}장 · 묶음 {sel.sequence.nunique()} · 카메라 {sel.model.nunique()} → {z} ({Path(z).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
