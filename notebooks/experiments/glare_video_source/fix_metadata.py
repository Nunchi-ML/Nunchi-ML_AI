"""촬영 시각 · 방향 보정 중 눈으로 확인한 것만 반영 (check_clock.py 다음에 실행).

check_clock.py 는 밝기로 낮/밤만 보므로 0 · −9시간 후보가 둘 다 낮이면 틀릴 수 있다.
사진 속 해 위치를 계산 위치와 대 보며 확인한 것:

1. 2019-02-23(UTC) iPhone 블랙박스 5묶음 (87p8 · VojW · YmcF · 8oLK · ZlXT, 같은 차)
   기록상 오전 7시(해 높이 1~3°, 해가 등 뒤) ↔ 사진엔 해가 정면 위에 뜸.
   +9시간(오후 4시)이면 해 높이 21~23° · 정면 Δ 9~17° → 해가 보이는 13장의 화면 가로 위치 오차 중앙값 0.12 (0시간이면 1.11)
2. 방향(heading) 기록 오류: 신호 대기 · 서행 중 기록 방향이 엉뚱함 (예: 87p8 131°(남동) ↔ 해가 화면 가운데).
   GPS 이동 경로 방향(앞뒤 15 m 지점을 잇는 방향)과 45° 넘게 다르면 heading_suspect.
   눈으로 확인한 오류 13장은 모두 걸리고(차이 57~173°), 맞는 사진은 0~3° 차이

(사진에서 해를 자동으로 찾는 방법은 나뭇가지 사이 하늘 · 흰 간판을 해로 잘못 잡아서 쓰지 않음)

    python notebooks/experiments/glare_video_source/fix_metadata.py
결과: candidates.csv · label_set.csv · dashcam_set.csv 의 offset_h · 시각 · 해 위치 · gap · front_low 다시 계산,
      track(경로 방향) · heading_suspect 열 추가
"""
import numpy as np
import pandas as pd

from nunchi_ml.sun import heading_gap, is_front_low, sun_position

import find_sun_frames as F

OUT = F.OUT
SEQ_OFFSET = {s: 9 for s in ("87p8mq0wqegtlwikolwypr", "VojW4b6GRzSZfP6MxUCklQ", "YmcFi26DSSOhShv6mwjmzw",
                             "8oLKqixIROa341kfNU9zrQ", "ZlXTCpLtTay1dwGxXff7OA")}
TRACK_M, HEADING_TOL = 15, 45


def track_bearing(g):
    """묶음 안 사진마다 앞뒤로 TRACK_M 이상 떨어진 지점을 잇는 이동 방향 (북=0). 거의 안 움직였으면 nan."""
    lat = np.radians(g.lat.to_numpy())
    x = 6371000 * np.radians(g.lon.to_numpy()) * np.cos(lat.mean())
    y = 6371000 * lat
    n, out = len(g), np.full(len(g), np.nan)
    for i in range(n):
        j = k = i
        while j > 0 and np.hypot(x[i] - x[j], y[i] - y[j]) < TRACK_M:
            j -= 1
        while k < n - 1 and np.hypot(x[k] - x[i], y[k] - y[i]) < TRACK_M:
            k += 1
        if np.hypot(x[k] - x[j], y[k] - y[j]) >= TRACK_M:
            out[i] = np.degrees(np.arctan2(x[k] - x[j], y[k] - y[j])) % 360
    return pd.Series(out, index=g.index)


def main():
    c = pd.read_csv(OUT / "candidates.csv").sort_values(["sequence", "captured_at_ms"])
    c["offset_h"] = c.sequence.map(SEQ_OFFSET).fillna(c.offset_h)
    c["clock_ok"] = c.offset_h.notna()
    c["track"] = c.groupby("sequence", group_keys=False)[["lat", "lon"]].apply(track_bearing).round(1)
    c["heading_suspect"] = np.nan_to_num(np.asarray(heading_gap(c.heading, c.track), float)) > HEADING_TOL
    keep = ["offset_h", "captured_at_ms", "clock_ok", "track", "heading_suspect"]
    for name in ("candidates.csv", "label_set.csv", "dashcam_set.csv"):
        if not (OUT / name).exists():                                 # 새 도로 폴더엔 라벨 세트가 아직 없음
            continue
        d = c if name == "candidates.csv" else pd.read_csv(OUT / name)
        d = d[[k for k in d.columns if k not in keep]].merge(c[["id"] + keep], on="id", how="left")
        t = (pd.to_datetime(d.captured_at_ms, unit="ms", utc=True)
             + pd.to_timedelta(d.offset_h.fillna(0), unit="h")).dt.tz_convert("Asia/Seoul")
        p = sun_position(t, d.lat, d.lon)
        d["captured_at_kst"] = t.dt.strftime("%Y-%m-%d %H:%M:%S")
        d["sun_alt"], d["sun_az"] = p.altitude.round(2).to_numpy(), p.azimuth.round(2).to_numpy()
        d["gap"] = heading_gap(d.sun_az, d.heading).round(1)
        d["front_low"] = is_front_low(d.sun_alt, d.gap) & d.clock_ok & ~d.heading_suspect
        d.to_csv(OUT / name, index=False, encoding="utf-8-sig")
        print(f"{name}: 시각 +9h {int((d.offset_h == 9).sum())}장 · 방향 의심 {int(d.heading_suspect.sum())}장 · 정면 저각도 {int(d.front_low.sum())}장")


if __name__ == "__main__":
    main()
