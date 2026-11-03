"""AI Hub 「특수환경 자율주행 3D 이미지」 Validation 역광 주행 → 라벨 세트 (약 300장).

사진을 눈으로 본 결과(대전 전 주행 정면 역광, 서울 drive_1189 해 정면, 나머지 서울은 건물에 가린 역광,
창원은 해가 산 뒤)에 맞춰 주행별로 간격을 다르게 뽑는다. 연속 프레임이라 일정 간격으로만.

    python notebooks/experiments/glare_video_source/make_aihub_label_set.py
먼저 aihub_index.py (meta.csv · low_sun_drives.csv) 와 원천 zip 풀기(images/<지역>/) 가 되어 있어야 함.
결과 (data/aihub_special/, git 제외):
    aihub_label_set.csv           분석용: 주행 · 해 높이 · 방위 · 카메라 화각
    label_pack/ · label_pack.zip  라벨러용: 사진 + 무작위 순서 label_sheet_blind.csv + label_tool.html
"""
import math
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
OUT = HERE.parents[2] / "data" / "aihub_special"
IMAGES = OUT / "images"

# (지역, 주행 이름 or None=그 지역 나머지 전부) → 몇 장마다 1장
STEP = [
    ("daejeon", None, 8),            # 전 주행 정면 역광 · 해 번짐 → 주력
    ("seoul2", "drive_1189", 8),     # 해가 화면 정면 → 심한 눈부심 후보
    ("seoul2", None, 100),           # 해는 건물 뒤 · 하늘만 밝음 → 0 ↔ 1 경계 사례
    ("changwon", None, 10_000),      # 해 산 뒤 · 노을 → 주행마다 1장 (0 예시)
]


def fov(calib):
    """calib.txt 의 intrinsicCameraParameter (fx, fy, cx, cy, ...) → 가로 · 세로 화각 (°)."""
    lines = calib.read_text(encoding="utf-8", errors="replace").splitlines()
    fx, fy = map(float, lines[lines.index("intrinsicCameraParameter") + 1].split(",")[:2])
    return 2 * math.degrees(math.atan(960 / fx)), 2 * math.degrees(math.atan(600 / fy))   # 1920 × 1200


def pick():
    drives = pd.read_csv(OUT / "low_sun_drives.csv", index_col=0)
    drives = drives[drives.split == "Validation"]
    drives["name"] = drives.index.str.rsplit("/", n=1).str[-1]
    rows, used = [], set()
    for area, name, step in STEP:
        for d in sorted((IMAGES / area).rglob("image_0")):
            key = (area, d.parent.name)
            if key in used or (name and d.parent.name != name):
                continue
            used.add(key)
            imgs = sorted(d.glob("*.jpg"))
            m = drives[(drives.area == area) & (drives.name == d.parent.name)].iloc[0]
            hfov, vfov = fov(d.parent / "calib.txt")
            for k in range(0, len(imgs), step):
                p = imgs[k]
                rows.append({"id": f"aihub:{area}/{d.parent.name}/{p.stem}",
                             "file": "images/" + p.relative_to(IMAGES).as_posix(),
                             "group": f"aihub_{area}", "sequence": f"{area}/{d.parent.name}",
                             "frame": k, "frames_in_drive": len(imgs), "captured_at_kst": m.time,
                             "lat": m.lat, "lon": m.lon, "weather_code": m.w,
                             "sun_alt": m.alt, "sun_az": m.az, "hfov": round(hfov, 1), "vfov": round(vfov, 1)})
    return pd.DataFrame(rows)


def main():
    sel = pick()
    sel.to_csv(OUT / "aihub_label_set.csv", index=False, encoding="utf-8-sig")
    print(sel.groupby("group").size().to_string(), f"\n합계 {len(sel)}장 · 주행 {sel.sequence.nunique()}개")

    p = OUT / "label_pack"
    shutil.rmtree(p, ignore_errors=True)
    for f in sel.file:
        (p / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT / f, p / f)
    blind = sel[["id", "file", "sequence"]].sample(frac=1, random_state=7).reset_index(drop=True)
    blind.insert(0, "no", range(1, len(blind) + 1))
    blind.assign(label="", confidence="", note="").to_csv(p / "label_sheet_blind.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(HERE / "label_tool.html", p / "label_tool.html")
    (p / "출처.txt").write_text("AI Hub 「특수환경 자율주행 3D 이미지」 (NIA, dataSetSn=198) Validation 원천 일부. "
                               "AI Hub 이용약관에 따라 팀 내부 라벨 작업용으로만 사용 · 외부 공유 금지.\n", encoding="utf-8")
    z = shutil.make_archive(str(OUT / "label_pack"), "zip", p)
    print(f"→ {z} ({Path(z).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
