"""새 카메라 독립 시험용 — 영상(YouTube CC BY 재사용 허용)에서 프레임을 뽑아 블라인드 라벨링 묶음을 만든다.

영상에는 GPS · 시각 · 방향이 없으므로 이 시험은 **사진만 보는 모델 vs 사진만 보는 규칙(밝기 규칙)** 비교다 (G 비교 아님).
편향을 피하려고 프레임은 ML 점수와 무관하게 **시간 간격(균등)으로만** 고른다.

    python notebooks/experiments/sun_visible_probe/new_camera_frames.py data/new_cameras --every 8 --max-per-video 60
결과 (data/new_cameras/, git 제외):
    label_pack/            라벨러에게 줄 폴더 (사진 + 무작위 순서 label_sheet_blind.csv + label_tool.html)
    private_frame_map.csv  분석자 전용: 익명 id → 영상 id · 시각(초)   (라벨러에게 주지 않는다)
라벨 규칙은 분석계획서 「라벨 기준표」 (해나 해 번짐이 화면에 보여야 1 이상, 식별 불능이면 2).
영상 저작권: 영상별 출처 · 작성자 · 라이선스는 data/new_cameras/attribution.csv (분석자용, 라벨 묶음에는 안 넣음). 내부 평가용, 재배포 안 함.
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "glare_video_source" / "label_tool.html"


def frames(video, every, cap):
    vc = cv2.VideoCapture(str(video))
    fps = vc.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps if n > 0 else 0
    ts = np.arange(3.0, max(dur - 1.0, 3.0), every)          # 앞 3초(인트로) 제외
    if len(ts) > cap:
        ts = ts[np.linspace(0, len(ts) - 1, cap).astype(int)]
    for t in ts:
        vc.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, im = vc.read()
        if ok and im is not None:
            yield float(t), im
    vc.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--every", type=float, default=8.0, help="프레임 간격(초)")
    ap.add_argument("--max-per-video", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pack = a.root / "label_pack"
    if pack.exists():
        raise FileExistsError("Existing label_pack must be preserved; never regenerate IDs or overwrite answers")
    (pack / "images").mkdir(parents=True)
    rows, used = [], set()
    for vdir in sorted(p for p in a.root.iterdir() if p.is_dir() and p.name != "label_pack"):
        vids = sorted(v for v in vdir.glob("video.*") if v.suffix in (".mp4", ".webm", ".mkv"))
        if not vids:
            continue
        seq = f"s{len(set(r['video'] for r in rows)) + 1:02d}"
        n = 0
        for t, im in frames(vids[0], a.every, a.max_per_video):
            h, w = im.shape[:2]
            if w > 1280:
                im = cv2.resize(im, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
            while True:
                fid = str(rng.randrange(10 ** 11, 10 ** 12))
                if fid not in used:
                    used.add(fid)
                    break
            path = pack / "images" / seq / f"{fid}.jpg"
            path.parent.mkdir(exist_ok=True)
            cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(path))
            rows.append({"id": fid, "file": f"images/{seq}/{fid}.jpg", "sequence": seq, "video": vdir.name, "t": round(t, 1)})
            n += 1
        print(f"{vdir.name}: {n}장 ({seq})")
    rng.shuffle(rows)
    with open(pack / "label_sheet_blind.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["no", "id", "file", "sequence", "label", "confidence", "note"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["id"], r["file"], r["sequence"], "", "", ""])
    with open(a.root / "private_frame_map.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sequence", "video", "t"])
        for r in rows:
            w.writerow([r["id"], r["sequence"], r["video"], r["t"]])
    blind = [dict(no=i, id=r["id"], file=r["file"], sequence=r["sequence"]) for i, r in enumerate(rows, 1)]
    html = TOOL.read_text(encoding="utf-8").replace(
        '<script id="embeddedRows" type="application/json">[]</script>',
        '<script id="embeddedRows" type="application/json">' + json.dumps(blind, ensure_ascii=True) + '</script>')
    (pack / "label_tool.html").write_text(html, encoding="utf-8")
    # 영상 제목(예: "역광 영상")이 라벨에 힌트가 되므로 출처표는 라벨 묶음에 넣지 않는다 (분석자 폴더에만 둠).
    print(f"총 {len(rows)}장 → {pack} (영상 {len(set(r['video'] for r in rows))}개). 라벨러에게는 label_pack 폴더만 준다.")


if __name__ == "__main__":
    main()
