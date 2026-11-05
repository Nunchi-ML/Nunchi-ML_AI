"""Non-destructive SD frame triage, temporal thinning and conservative dedup.

No labels inferred, no training, no brightness/sharpness-based hard rejection.
All original images and source tables remain untouched. Private outputs only.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABELS = ["glare_label_rater1", "glare_label_rater2", "glare_label_final", "tunnel_state", "usable", "notes"]
PRIORITY = {"INF": 0, "EVENT": 1, "USER": 2, "PARK": 3}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_image(root, relative):
    root = Path(root).resolve()
    path = (root/str(relative)).resolve()
    if not path.is_relative_to(root) or not str(relative).replace("\\", "/").startswith("images/"):
        raise ValueError("Source image path must remain inside root/images")
    return path


def clip_time(clip, second):
    try:
        return datetime.strptime(str(clip)[:15], "%Y%m%d_%H%M%S") + timedelta(seconds=float(second))
    except ValueError:
        return None


def perceptual(gray):
    # Crop out most timestamp/hood pixels. Low-resolution proxies only.
    height, width = gray.shape
    roi = gray[int(height*.08):int(height*.82), int(width*.04):int(width*.96)]
    thumb = cv2.resize(roi, (32, 18), interpolation=cv2.INTER_AREA)
    small = cv2.resize(roi, (9, 8), interpolation=cv2.INTER_AREA)
    dh = 0
    for bit in (small[:, 1:] > small[:, :-1]).ravel():
        dh = (dh << 1) | int(bit)
    dct = cv2.dct(cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8]
    threshold = np.median(dct.ravel()[1:])
    ph = 0
    for bit in (dct > threshold).ravel():
        ph = (ph << 1) | int(bit)
    return thumb, dh, ph


def inspect_one(args):
    root, row = args
    path = safe_image(root, row["image"])
    result = dict(image=row["image"], exists=path.is_file(), decode_ok=False)
    if not path.is_file():
        return result, None
    stat = path.stat()
    payload = path.read_bytes()
    result.update(bytes=len(payload), source_mtime_ns=stat.st_mtime_ns, sha256=hashlib.sha256(payload).hexdigest())
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
    if image is None or min(image.shape[:2]) < 8:
        return result, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    road = gray[int(h*.4):int(h*.82), int(w*.1):int(w*.9)]
    thumb, dh, ph = perceptual(gray)
    result.update(decode_ok=True, analysis_width=w, analysis_height=h,
                  reduced_mean=float(gray.mean()), reduced_white_ratio=float((gray >= 250).mean()),
                  reduced_dark_ratio=float((gray <= 15).mean()), road_std=float(road.std()),
                  road_laplacian_var=float(cv2.Laplacian(road, cv2.CV_32F).var()),
                  dhash=f"{dh:016x}", phash=f"{ph:016x}")
    return result, thumb


def exact_representatives(frame, old_hashes):
    valid = frame[frame.decode_ok].copy()
    valid["category_priority"] = valid.source_category.map(PRIORITY).fillna(9)
    valid = valid.sort_values(["category_priority", "image"], kind="stable")
    canonical = {}
    for row in valid.itertuples():
        canonical.setdefault(row.sha256, row.image)
    output = frame.copy()
    output["exact_representative"] = output.sha256.map(canonical).fillna("")
    output["existing_reference_duplicate"] = output.sha256.isin(old_hashes) & output.decode_ok
    output["exact_duplicate"] = output.decode_ok & output.image.ne(output.exact_representative)
    return output


def candidate_reasons(frame, transitions, interval=10):
    if interval < 1:
        raise ValueError("Positive sampling interval required")
    reasons = defaultdict(set)
    for _, clip in frame[frame.decode_ok & frame.source_category.ne("PARK")].groupby("clip_id", sort=True):
        clip = clip.sort_values(["second", "image"])
        first, last = clip.iloc[0], clip.iloc[-1]
        reasons[first.image].add("clip_start")
        reasons[last.image].add("clip_end")
        for row in clip.itertuples():
            if float(row.second) % interval == 0:
                reasons[row.image].add("temporal_representative")
        if clip.mean_brightness.max()-clip.mean_brightness.min() >= 25:
            reasons[clip.loc[clip.mean_brightness.idxmin(), "image"]].add("brightness_range_extreme")
            reasons[clip.loc[clip.mean_brightness.idxmax(), "image"]].add("brightness_range_extreme")
        if clip.white_ratio.max() >= .10:
            reasons[clip.loc[clip.white_ratio.idxmax(), "image"]].add("high_white_review")
        weak = clip[clip.road_laplacian_var.lt(20)]
        if len(weak):
            reasons[weak.loc[weak.road_laplacian_var.idxmin(), "image"]].add("low_sharpness_review")
    # These are pre-existing brightness-change candidates, not tunnel labels.
    known = set(frame.image)
    for row in transitions.itertuples():
        for column in ["before_image", "after_image"]:
            path = str(getattr(row, column))
            if path not in known:
                raise ValueError("Transition references an unknown source image")
            reasons[path].add("transition_context")
        neighborhood = frame[frame.clip_id.eq(row.clip_id) & frame.second.between(row.second-1, row.second+1)]
        for path in neighborhood.image:
            reasons[path].add("transition_context")
    return reasons


def is_near(row, anchor, thumb, other):
    # No discard from similarity alone across distant time/different dates.
    if row["group_id"] != anchor["group_id"]:
        return False, None
    a, b = row.get("filename_datetime"), anchor.get("filename_datetime")
    if a is None or b is None or pd.isna(a) or pd.isna(b) or abs((a-b).total_seconds()) > 120:
        return False, None
    if (int(row["phash"], 16)^int(anchor["phash"], 16)).bit_count() > 4:
        return False, None
    if (int(row["dhash"], 16)^int(anchor["dhash"], 16)).bit_count() > 4:
        return False, None
    distance = float(np.mean(abs(thumb.astype(np.float32)-other.astype(np.float32))))
    return distance <= 3., distance


def thin_candidates(frame, thumbs, reasons):
    records = frame.to_dict("records")
    buckets, anchors = defaultdict(list), {}
    protected = {"transition_context", "brightness_range_extreme", "high_white_review", "low_sharpness_review"}
    for row in records:
        row["candidate_reasons"] = "|".join(sorted(reasons.get(row["image"], set())))
        row["near_representative"], row["near_thumb_mae"] = "", None
        row["leakage_group_provisional"] = "SD:date:"+str(row["group_id"])
        if not row["decode_ok"]:
            row["decision"] = "exclude_missing_or_corrupt"
        elif row["existing_reference_duplicate"]:
            row["decision"] = "exclude_existing_exact_duplicate"
        elif row["exact_duplicate"]:
            row["decision"] = "exclude_exact_duplicate"
        elif row["source_category"] == "PARK":
            row["decision"] = "separate_parking"
        elif not row["candidate_reasons"]:
            row["decision"] = "reserve_temporal_redundant"
        else:
            row["decision"] = "retain_candidate"
    candidates = [r for r in records if r["decision"] == "retain_candidate"]
    # Keep hard/change contexts before selecting ordinary representatives.
    candidates.sort(key=lambda r: (not bool(reasons[r["image"]]&protected), PRIORITY.get(r["source_category"], 9), r["image"]))
    matches = []
    for row in candidates:
        ph = int(row["phash"], 16)
        keys = [(str(row["group_id"]), bit, (ph >> (bit*8)) & 255) for bit in range(8)]
        possible = set(i for key in keys for i in buckets[key])
        for image in sorted(possible):
            anchor = anchors[image]
            match, distance = is_near(row, anchor, thumbs[row["image"]], thumbs[image])
            if match:
                row["near_representative"], row["near_thumb_mae"] = image, distance
                matches.append(dict(image=row["image"], representative=image, thumb_mae=distance,
                                    same_clip=row["clip_id"] == anchor["clip_id"],
                                    protected_context=bool(reasons[row["image"]]&protected)))
                if not reasons[row["image"]]&protected:
                    row["decision"] = "reserve_near_duplicate"
                break
        if row["decision"] == "retain_candidate":
            anchors[row["image"]] = row
            for key in keys:
                buckets[key].append(row["image"])
    return pd.DataFrame(records), matches


def review_batch(candidates, size=300):
    """Diverse development review sample; never a final test selection."""
    if size < 1:
        raise ValueError("Positive review size required")
    candidates = candidates.copy()
    if not len(candidates):
        return candidates
    candidates["illumination_proxy"] = pd.cut(candidates.mean_brightness, [-np.inf, 45, 110, np.inf], labels=["low", "middle", "high"]).astype(str)
    strata = defaultdict(list)
    for row in candidates.to_dict("records"):
        key = (str(row["group_id"]), str(row["source_category"]), row["illumination_proxy"])
        strata[key].append(row)
    for rows in strata.values():
        rows.sort(key=lambda r: hashlib.sha256(("47:"+r["image"]).encode()).hexdigest())
    selected, used, clip_counts = [], set(), Counter()
    for cap in [1, 2, 3, len(candidates)]:
        while len(selected) < min(size, len(candidates)):
            changed = False
            for key in sorted(strata):
                eligible = next((r for r in strata[key] if r["image"] not in used and clip_counts[r["clip_id"]] < cap), None)
                if eligible is not None:
                    selected.append(eligible); used.add(eligible["image"]); clip_counts[eligible["clip_id"]] += 1
                    changed = True
                    if len(selected) >= min(size, len(candidates)):
                        break
            if not changed:
                break
        if len(selected) >= min(size, len(candidates)):
            break
    return pd.DataFrame(selected)


def gallery(path, frame, root, title):
    rows = []
    for row in frame.to_dict("records"):
        image = safe_image(root, row["image"])
        rows.append(dict(src=quote(os.path.relpath(image, path.parent).replace("\\", "/"), safe="/"),
                         image=row["image"], clip=row["clip_id"], second=float(row["second"]),
                         category=row["source_category"], reasons=row.get("candidate_reasons", ""),
                         flags=row.get("quality_flags", "")))
    data = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    body = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SD frame review</title><style>body{font:16px sans-serif;margin:24px;background:#f6f7fa;color:#18202c}button,select{font:inherit;margin:4px;padding:8px}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}article{background:white;padding:10px;border:1px solid #ccc;border-radius:8px}img{width:100%;height:auto}small{overflow-wrap:anywhere;display:block}p{max-width:1000px}</style>
<h1>TITLE</h1><p>원본을 보존한 분석용 선별입니다. 밝기/선명도 표시는 품질 검토 단서이며 눈부심·초점불량·야간·터널의 확정 라벨이 아닙니다. 이 목록은 개발 검토용이며 최종 시험 분할이 아닙니다.</p>
<label>유형 <select id="category"><option value="">전체</option><option>INF</option><option>EVENT</option><option>USER</option><option>PARK</option></select></label>
<button id="prev">이전</button><button id="next">다음</button><span id="count"></span><div id="grid"></div>
<script>const rows=DATA;let page=0;const size=24;const grid=document.getElementById('grid'),sel=document.getElementById('category');
function render(){const a=rows.filter(r=>!sel.value||r.category===sel.value);page=Math.max(0,Math.min(page,Math.ceil(a.length/size)-1));grid.replaceChildren();for(const r of a.slice(page*size,(page+1)*size)){const card=document.createElement('article'),img=document.createElement('img');img.loading='lazy';img.src=r.src;img.alt='검토용 원본 프레임';const caption=document.createElement('small');caption.textContent=r.clip+' / '+r.second+'초 / '+r.category+' / '+r.flags;const why=document.createElement('small');why.textContent='선별 이유: '+r.reasons;const link=document.createElement('a');link.href=r.src;link.target='_blank';link.textContent='원본 열기';card.append(img,caption,why,link);grid.append(card);}document.getElementById('count').textContent=a.length+'장 · '+(a.length?page+1:0)+'/'+Math.ceil(a.length/size)+'페이지';}
document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};sel.onchange=()=>{page=0;render()};render();</script></html>'''
    path.write_text(body.replace("TITLE", html.escape(title)).replace("DATA", data), encoding="utf-8")


def run(root, data, out, workers=4, interval=10, review_size=300):
    root, data, out = Path(root).resolve(), Path(data).resolve(), Path(out).resolve()
    if out.exists() or out.parent != data or out.is_relative_to(root):
        raise ValueError("Use a new private output directly under AI data, outside source")
    if workers < 1 or workers > 8:
        raise ValueError("Use1..8local threads")
    cv2.setNumThreads(1)
    tables = [root/n for n in ["frames_and_labels.csv", "video_inventory.csv", "model3_brightness_transition_candidates.csv", "README.txt", "summary.json"]]
    reference_table = data/"road_visibility_20261104/run30/training_features.csv"
    sources = tables+[reference_table, Path(__file__)]
    hashes = {str(p): sha(p) for p in sources}
    frame = pd.read_csv(tables[0], dtype={"group_id": str})
    transitions = pd.read_csv(tables[2])
    inventory = pd.read_csv(tables[1])
    if (not frame.image.is_unique or frame.duplicated(["clip_id", "second"]).any()
            or not set(frame.clip_id).issubset(set(inventory.file.str.replace(r"\.[^.]+$", "", regex=True)))):
        raise ValueError("Unambiguous source image/clip identity required")
    original_files = sorted(p.relative_to(root).as_posix() for p in (root/"images").rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if set(original_files) != set(frame.image):
        raise ValueError("Manifest and original image set must match exactly")
    old_hashes = set(pd.read_csv(reference_table, usecols=["image_sha256"]).image_sha256)
    old_pack = data/"new_cameras/label_pack/images"
    old_images = sorted(p for p in old_pack.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    old_photo_hashes = {str(p): sha(p) for p in old_images}
    old_hashes.update(old_photo_hashes.values())
    out.mkdir()
    specification = dict(status="frozen_before_scan", purpose="Reversible reduction for human development review; no labels/training/deletion",
        original_photos=len(frame), clips=frame.clip_id.nunique(), date_groups=frame.group_id.nunique(),
        preservation="Never write source images/tables. SHA256alloriginals, metadata before/after, reference hashes preserved. Copy only300review images without reencoding.",
        filters="Missing/corrupt and exact duplicates exclude from primary; PARK separate reserve. No darkness/blur/whiteout hard exclusion.",
        temporal=dict(seconds=interval, keep="clipfirst/last, brightnessrange>=25extremes, maxwhite>=.10, minroadLaplacian<20, all229sourcebrightnesschange contexts +/-1sec"),
        near_duplicate="Among selected candidates only: same filename-date, <=120filename-seconds, pHash<=4,dHash<=4,32x18ROI gray MAE<=3. Eight8bit hash buckets. Preserve hard/change contexts even when near; record links. No distant/crossdate perceptual discard.",
        quality="ReducedJPEGdecode1/4/ROI metrics only, not calibrated luminance/human impairment or defocus/tunnel labels. Filename timing unverified; all pre-existing brightness transition candidates retained separately.",
        review=dict(size=review_size, strata="date/category/brightnessproxy(<45,45..110,>110), deterministicSHA seed47, roundrobin/clipdiversity", purpose="Development review only; not representative prevalence/heldout performance"),
        splitting="Provisional filename-date groups; never split adjacentclips/overlaps byframe. No final train/test split, no geometry imputation; location/heading unavailable in supplied tables.",
        input_sha256=hashes, reference_photo_count=len(old_images), source_root=str(root),
        versions={n: importlib.metadata.version(n) for n in ["numpy", "pandas", "opencv-python-headless"]})
    (out/"specification_before_run.json").write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, out/"source_before_run.py")
    start = time.monotonic(); features, thumbs = [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, (result, thumb) in enumerate(pool.map(inspect_one, ((root, row) for row in frame.to_dict("records"))), 1):
            features.append(result)
            if thumb is not None:
                thumbs[result["image"]] = thumb
            if count % 1000 == 0:
                print("SD audit scanned", count, "/", len(frame), flush=True)
    metrics = pd.DataFrame(features)
    for column in ["sha256", "bytes", "source_mtime_ns", "road_laplacian_var", "phash", "dhash"]:
        if column not in metrics:
            metrics[column] = np.nan
    frame = frame.merge(metrics, on="image", validate="one_to_one")
    frame["filename_datetime"] = [clip_time(c, t) for c, t in zip(frame.clip_id, frame.second)]
    frame["stable_id"] = ["sd_"+hashlib.sha256(s.encode()).hexdigest()[:16] for s in frame.image]
    if not frame.stable_id.is_unique:
        raise ValueError("Stable review ID collision")
    def flags(row):
        values = []
        if row.mean_brightness < 45: values.append("low_brightness_review")
        if row.white_ratio >= .10: values.append("high_white_review")
        if row.road_laplacian_var < 20: values.append("low_sharpness_review")
        if str(row.source_category) == "USER": values.append("scene_review")
        return "|".join(values)
    frame["quality_flags"] = [flags(r) for r in frame.itertuples()]
    frame = exact_representatives(frame, old_hashes)
    reasons = candidate_reasons(frame, transitions, interval)
    frame, matches = thin_candidates(frame, thumbs, reasons)
    selected = frame[frame.decision.eq("retain_candidate")].copy()
    review = review_batch(selected, review_size)
    parking = frame[frame.source_category.eq("PARK")].copy()
    parking_review = parking[parking.decode_ok & ~parking.exact_duplicate & ~parking.existing_reference_duplicate].sort_values(["clip_id", "second"]).groupby("clip_id", sort=True).head(1)
    for filename, part in [("all_frame_decisions.csv", frame), ("curated_candidates.csv", selected),
                           ("parking_reserve.csv", parking), ("parking_representatives.csv", parking_review), ("review_first300.csv", review)]:
        part.to_csv(out/filename, index=False, encoding="utf-8-sig")
    pd.DataFrame(matches, columns=["image", "representative", "thumb_mae", "same_clip", "protected_context"]).to_csv(out/"near_duplicate_pairs.csv", index=False, encoding="utf-8-sig")
    shutil.copyfile(tables[2], out/"model3_transition_candidates_original.csv")
    transitions["before_decision"] = transitions.before_image.map(frame.set_index("image").decision)
    transitions["after_decision"] = transitions.after_image.map(frame.set_index("image").decision)
    transitions.to_csv(out/"model3_transition_preservation.csv", index=False, encoding="utf-8-sig")
    counts = frame.groupby(["group_id", "source_category", "decision"]).size().rename("count").reset_index()
    counts.to_csv(out/"counts_by_date_category.csv", index=False, encoding="utf-8-sig")
    pack = out/"review_first300"; (pack/"images").mkdir(parents=True)
    labels = []
    for row in review.to_dict("records"):
        dest = pack/"images"/(row["stable_id"]+".jpg")
        shutil.copyfile(safe_image(root, row["image"]), dest)
        if sha(dest) != row["sha256"]:
            raise ValueError("Review copy image bytes changed")
        labels.append(dict(id=row["stable_id"], image="images/"+dest.name, group_id=row["group_id"], clip_id=row["clip_id"], second=row["second"],
                           **{column:row.get(column, "") for column in LABELS}))
    pd.DataFrame(labels).to_csv(pack/"labels_blank.csv", index=False, encoding="utf-8-sig")
    gallery(out/"review_first300.html", review, root, "SD 1차 검토용 선별300장")
    gallery(out/"curated_candidates.html", selected, root, "SD 확장 검토 후보")
    gallery(out/"parking_representatives.html", parking_review, root, "주차 자료 별도 검토 대표")
    # A portable300-image pack references only byte-identical local copies.
    portable = review.copy()
    portable["image"] = ["images/"+s+".jpg" for s in portable.stable_id]
    gallery(pack/"review.html", portable, pack, "SD 1차 검토용300장 원본복사 묶음")
    final_files = sorted(p.relative_to(root).as_posix() for p in (root/"images").rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if final_files != original_files:
        raise ValueError("Source image set changed")
    def check(row):
        path = safe_image(root, row["image"])
        return sha(path) == row["sha256"] if row["exists"] else not path.exists()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if not all(pool.map(check, frame.to_dict("records"))):
            raise ValueError("Source image bytes changed during curation")
    if any(sha(p) != h for p, h in (hashes|old_photo_hashes).items()):
        raise ValueError("Original tables/reference photos/source changed")
    labels_unchanged = all(frame[c].isna().all() for c in LABELS if c in frame)
    result = dict(status="complete_non_destructive_curation", original_photos=len(frame), original_clips=int(frame.clip_id.nunique()),
        date_groups=int(frame.group_id.nunique()), decisions=frame.decision.value_counts().to_dict(), selected_photos=len(selected),
        selected_clips=int(selected.clip_id.nunique()), review_photos=len(review), review_clips=int(review.clip_id.nunique()),
        parking_photos=len(parking), parking_representatives=len(parking_review), near_pairs=len(matches),
        preserved_model3_brightness_candidates=len(transitions), human_labels_supplied=not labels_unchanged,
        geometry_available=False, final_split_created=False, model_training=False, source_files_deleted=0,
        original_image_sha256={r.image:r.sha256 for r in frame[frame.exists].itertuples()}, input_sha256=hashes,
        reference_photo_sha256=old_photo_hashes, elapsed_seconds=float(time.monotonic()-start),
        limits="This is reversible development triage, not semantic ground truth, optical diagnosis or a final test split. Brightness proxies can mix night/tunnel/shadow/glare; protected cases retained. Filename time/date unverified. Actual camera/route independence and labels still require review.")
    (out/"report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out/"report.html").write_text('<!doctype html><meta charset="utf-8"><h1>SD 원본 보존·분석용 선별</h1><p>원본 삭제·자동 라벨·모델 학습 없이 선별했습니다. 모든 프레임은 다시 포함할 수 있습니다.</p><p><a href="review_first300/review.html">1차300장 검토</a> · <a href="curated_candidates.html">확장 후보</a> · <a href="parking_representatives.html">주차 대표</a></p>'+pd.DataFrame([dict(decision=k,count=v) for k,v in result["decisions"].items()]).to_html(index=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ["original_image_sha256", "input_sha256", "reference_photo_sha256"]}, ensure_ascii=True, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--review-size", type=int, default=300)
    args = parser.parse_args(); run(args.root, args.data, args.out, args.workers, args.interval, args.review_size)
