"""Create a blind auxiliary-label bundle, never fill human answers from ML.

Sample by camera/capture group without existing labels or model predictions.
Share ONLY review_bundle/, not its parent containing the private ID mapping.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from g_visibility_probe import image_paths

FIELDS = [
    ["sun_disc", "해 원반", [["visible", "보임"], ["partly_occluded", "일부 가림"], ["not_visible", "안 보임"], ["unclear", "판단 어려움"]]],
    ["sun_occluder", "직접 관찰한 원반 가림", [["none", "가림 없음"], ["building", "건물"], ["tree", "나무"], ["vehicle", "차량"], ["cloud", "구름"], ["other", "기타"], ["unknown", "알 수 없음"]]],
    ["bloom", "빛 번짐", [["visible", "보임"], ["not_visible", "안 보임"], ["unclear", "판단 어려움"]]],
    *[[key, title, [["clear", "식별 가능"], ["reduced", "식별 저하"], ["not_assessable", "대상 없음·멀거나 판단 불가"]]]
      for key, title in [("lane_visibility", "차선"), ("vehicle_visibility", "앞차"), ("signal_visibility", "신호·표지판")]],
    ["degradation_cause", "주된 식별 저하 원인", [["none", "저하 없음"], ["glare", "눈부심·번짐"], ["motion_blur", "흔들림·초점"], ["occlusion", "다른 물체 가림"], ["darkness", "어두움"], ["other", "기타"], ["unclear", "판단 어려움"]]],
]
ANSWER_COLUMNS = [f[0] for f in FIELDS] + ["sun_x", "sun_y", "notes"]


def sample(snapshot, per_camera, seed):
    rng, selected = np.random.default_rng(seed), []
    for _, camera in snapshot.groupby("camera", sort=True):
        queues = [list(rng.permutation(group.index)) for _, group in camera.groupby("group", sort=True)]
        rng.shuffle(queues)
        chosen = []
        while len(chosen) < min(per_camera, len(camera)):
            for queue in queues:
                if queue and len(chosen) < per_camera:
                    chosen.append(queue.pop())
        selected.extend(chosen)
    return snapshot.loc[rng.permutation(selected)].copy()


def build(snapshot, data_root, out, per_camera=24, seed=47):
    if per_camera < 1 or not snapshot.id.is_unique:
        raise ValueError("Positive sample count and unique IDs required")
    selected = sample(snapshot, per_camera, seed)
    paths = image_paths(selected, data_root)
    bundle = out / "review_bundle"
    assets = bundle / "assets"
    if bundle.exists():
        raise FileExistsError("Review bundle already exists; use a new output directory to preserve answers")
    assets.mkdir(parents=True)
    mapping, records, image_digest = [], [], hashlib.sha256()
    for n, (row, path) in enumerate(zip(selected.itertuples(), paths), 1):
        key = f"R{n:04}"
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable frame: {path}")
        h, w = image.shape[:2]
        scale = min(1., 1280/max(h, w))
        thumb = cv2.resize(image, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
        # Re-encode full aspect ratio: anonymous filename, no source EXIF/time.
        ok, encoded = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("Cannot encode review image")
        (assets / f"{key}.jpg").write_bytes(encoded.tobytes())
        image_digest.update(key.encode())
        image_digest.update(encoded.tobytes())
        records.append(dict(review_id=key, image=f"assets/{key}.jpg", **dict.fromkeys(ANSWER_COLUMNS, "")))
        mapping.append(dict(review_id=key, id=row.id, camera=row.camera, group=row.group,
                            image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    anonymous = pd.DataFrame(records)
    anonymous.to_csv(bundle / "blank_annotations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mapping).to_csv(out / "private_review_mapping.csv", index=False, encoding="utf-8-sig")
    # Template receives anonymous IDs + options only, never original labels/G/metadata.
    payload = dict(records=[{k: d[k] for k in ("review_id", "image")} for d in records], fields=FIELDS,
                   columns=ANSWER_COLUMNS)
    serialized = json.dumps(payload, ensure_ascii=False)
    # R0001... filenames repeat across bundles: scope saved answers by image
    # content too, otherwise a different 96-image bundle could inherit labels.
    identity = hashlib.sha256(serialized.encode()+image_digest.digest()).hexdigest()
    template = Path(__file__).with_name("visibility_review.html").read_text(encoding="utf-8")
    (bundle / "review.html").write_text(template.replace("__DATA__", serialized).replace("__BUNDLE__", identity), encoding="utf-8")
    (out / "review_method.json").write_text(json.dumps(dict(seed=seed, per_camera=per_camera, rows=len(selected),
        groups=int(selected.group.nunique()), manifest=identity, sample_uses_labels=False,
        snapshot_sha256=hashlib.sha256(snapshot.to_csv(index=False).encode()).hexdigest()), indent=2), encoding="utf-8")
    return bundle, len(selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-camera", type=int, default=24)
    args = p.parse_args()
    bundle, count = build(pd.read_csv(args.snapshot, dtype={"id": str}), args.data_root, args.out, args.per_camera)
    print(f"Blind review: {count} photos -> {bundle / 'review.html'}; all answers blank")


if __name__ == "__main__":
    main()
