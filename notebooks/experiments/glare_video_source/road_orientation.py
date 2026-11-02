"""대구 큰길 후보들의 도로선을 받아 '동서 방향 비율'을 본다 (오후 · 아침 해를 정면으로 마주치려면 동서로 뻗은 길이 유리).

    python notebooks/experiments/glare_video_source/road_orientation.py 국채보상로 앞산순환로 ...
결과: data/roads/<도로 이름>.json (없을 때만 받음) · 도로별 총 길이 · 동서 방향(방위 60~120° · 240~300°) 비율
"""
import json
import math
import sys
import time

import fetch_road as R


def summary(lines):
    tot = ew = 0.0
    for l in lines:
        for (x0, y0), (x1, y1) in zip(l, l[1:]):
            dx = (x1 - x0) * 111320 * math.cos(math.radians(y0)); dy = (y1 - y0) * 110940
            L = math.hypot(dx, dy); b = math.degrees(math.atan2(dx, dy)) % 180
            tot += L; ew += L if 60 <= b <= 120 else 0
    return tot / 1000, ew / tot if tot else 0


def main():
    for name in sys.argv[1:]:
        f = R.ROOT / "data" / "roads" / f"{name}.json"
        if not f.exists():
            lines = R.fetch(name); time.sleep(2)
            if not lines:
                print(f"{name}: 없음"); continue
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"name": name, "lines": lines}, ensure_ascii=False), encoding="utf-8")
        km, ew = summary(json.loads(f.read_text(encoding="utf-8"))["lines"])
        print(f"{name}: 도로선 합계 {km:5.1f} km · 동서 방향 {ew:.0%}")


if __name__ == "__main__":
    main()
