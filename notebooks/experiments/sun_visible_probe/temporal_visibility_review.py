"""Prepare reference-only neighboring frames without changing labels or training.

All human50 rows are included. Priority reasons/old answers remain outside the
blind review folder. Future frames are annotation evidence, never model inputs.
This cannot validate correspondence between camera and in-person human vision.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import pandas as pd

OFFSETS = (-2., -1., 0., 1., 2.)
FIELDS = ["case_id", "rater", "scene", "sun_evidence", "lane", "vehicle", "sign",
          "same_target", "cause", "temporal_change", "proposed_grade", "unsure", "note"]
CHOICES = dict(scene=["front_road", "rear_road", "non_driving", "uncertain"],
               sun_evidence=["yes", "no", "uncertain"],
               lane=["identifiable", "difficult", "unidentifiable", "absent", "uncertain"],
               vehicle=["identifiable", "difficult", "unidentifiable", "absent", "uncertain"],
               sign=["identifiable", "difficult", "unidentifiable", "absent", "uncertain"],
               same_target=["yes", "no", "uncertain"],
               cause=["solar", "fog_rain", "blur_resolution", "occlusion", "viewpoint_motion", "exposure_possible", "mixed", "none", "uncertain"],
               temporal_change=["stable", "clear_to_impaired", "impaired_to_clear", "intermittent", "not_comparable", "uncertain"],
               proposed_grade=["0", "1", "2", "abstain"], unsure=["yes", "no"])
LABELS = dict(scene="장면", sun_evidence="태양·번짐 근거", lane="차선", vehicle="차량", sign="표지판",
              same_target="같은 대상 확인", cause="저하 원인", temporal_change="전후 변화", proposed_grade="새 보조 등급 (선택)", unsure="애매함")
VALUE_LABELS = dict(front_road="전방 도로", rear_road="후방 도로", non_driving="비주행 장면", uncertain="판단 어려움",
                    yes="있음/예", no="없음/아니오", identifiable="식별 가능", difficult="식별 어려움", unidentifiable="식별 불가",
                    absent="대상 없음", solar="태양광", fog_rain="안개·비", blur_resolution="흔들림·해상도", occlusion="가림",
                    viewpoint_motion="시점·대상 이동", exposure_possible="노출 변화 가능성", mixed="여러 원인", none="저하 없음",
                    stable="큰 변화 없음", clear_to_impaired="보임→저하", impaired_to_clear="저하→보임", intermittent="반복 변화",
                    not_comparable="직접 비교 어려움", abstain="판정 보류")


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def under(root, relative):
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes data root")
    return result


def match_rows(sheet, mapping, selected):
    for frame in (sheet, mapping, selected):
        if frame.id.isna().any() or not frame.id.is_unique:
            raise ValueError("IDs must be unique and present")
    if len(pd.Index(selected.id).difference(sheet.id)):
        raise ValueError("Selected ID missing from original blind sheet")
    rows = selected[["id"]].merge(sheet[["id", "file"]], on="id", how="left", validate="one_to_one").merge(
        mapping[["id", "video", "t"]], on="id", how="left", validate="one_to_one")
    if rows.video.isna().any() or not np.isfinite(rows.t.to_numpy(float)).all() or rows.t.lt(0).any():
        raise ValueError("Every selected row needs valid video/time mapping")
    return rows


def read_at(cap, seconds):
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid frame time")
    if not cap.set(cv2.CAP_PROP_POS_MSEC, seconds*1000):
        raise ValueError("Video seek failed")
    ok, image = cap.read()
    if not ok or image is None:
        raise ValueError(f"Cannot decode requested frame at {seconds}s")
    reported_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    if image.shape[1] > 1280:
        image = cv2.resize(image, (1280, int(image.shape[0]*1280/image.shape[1])), interpolation=cv2.INTER_AREA)
    # Decoder-reported position is diagnostic, not a guarantee of exact PTS.
    return image, dict(requested_seconds=seconds, decoder_reported_seconds=reported_ms/1000,
                       decoder_frame_number=frame_number, timestamp_note="OpenCV decoder report; not independently verified PTS")


def save_jpeg(path, image):
    ok, raw = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("JPEG encoding failed")
    path.write_bytes(raw.tobytes())


def render(cases):
    options = "".join('<label>'+LABELS[key]+'<select data-field="'+key+'"><option value=""></option>'+''.join('<option value="'+v+'">'+html.escape(VALUE_LABELS.get(v, v))+'</option>' for v in values)+'</select></label>' for key, values in CHOICES.items())
    payload = json.dumps(cases, ensure_ascii=True).replace("<", "\\u003c")
    return '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>전후 장면 식별성 검토</title>
<style>body{font:15px system-ui;margin:20px;background:#f5f6f8;color:#182334}button,input,select,textarea{font:inherit;padding:8px}button{cursor:pointer}header{max-width:1100px}nav{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:16px 0}.frames{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}.frames figure{margin:0;background:white;padding:8px;border:1px solid #ccd3df}.frames figure.center{border:3px solid #2469aa}.frames img{width:100%;height:auto}label{display:flex;flex-direction:column;gap:4px}.fields{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:18px}textarea{width:95%;height:70px}#status{min-height:24px}@media(max-width:800px){.frames{grid-template-columns:repeat(2,minmax(0,1fr))}.fields{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>
<header><h1>전후 장면으로 확인하는 정보 식별성</h1><p>가운데는 기존 정답을 붙인 원본 사진입니다. 전후 장면은 관찰 보조용이며 실제 사람의 현장 시야·휘도·적응시간을 측정하지 않습니다. 사진을 누르면 원본 크기로 열립니다.</p><p><b>대상이 원래 없거나 이동·가림으로 사라진 경우를 눈부심 손실로 세지 마세요.</b> 같은 대상·장면인지 확인하고 태양광/노출/흔들림/안개를 구분하세요. 새 보조 등급은 선택 항목입니다. 기존 정답을 덮어쓰지 않습니다. 미래 프레임은 모델 입력으로 사용하지 않습니다.</p></header>
<nav><label>판정자 <input id="rater" autocomplete="off" placeholder="이름 또는 AI 판정 버전"></label><button id="apply">판정자 적용</button><button id="prev">이전</button><strong id="counter"></strong><button id="next">다음</button><button id="download">새 보조 판정 CSV 저장</button><span id="status" role="status"></span></nav>
<div id="frames" class="frames"></div><div class="fields">'''+options+'''</div><label>관찰 근거·같은 대상 확인·한계<textarea id="note"></textarea></label>
<script>
const cases='''+payload+''';let index=0,answers={},activeRater='';
const fields='''+json.dumps(FIELDS)+''';
const rater=document.getElementById('rater'),status=document.getElementById('status');
const key=()=> 'temporal-visibility-v2:'+cases.map(c=>c.case_id+':'+c.center_sha256).join(',')+':'+activeRater;
function save(){if(!activeRater||activeRater!==rater.value.trim()){status.textContent='판정자 이름을 적고 판정자 적용을 누르세요.';return false;}const a={case_id:cases[index].case_id,rater:activeRater,note:document.getElementById('note').value};document.querySelectorAll('[data-field]').forEach(e=>a[e.dataset.field]=e.value);answers[a.case_id]=a;try{localStorage.setItem(key(),JSON.stringify(answers));status.textContent='이 판정자 이름으로 브라우저에 저장됨';}catch(e){status.textContent='브라우저 저장 불가: CSV로 내보내세요.';}return true;}
function show(){const c=cases[index];document.getElementById('counter').textContent=c.case_id+' · '+(index+1)+' / '+cases.length;const box=document.getElementById('frames');box.replaceChildren();for(const f of c.frames){const fig=document.createElement('figure');if(f.offset===0)fig.className='center';const a=document.createElement('a');a.href=f.file;a.target='_blank';a.rel='noopener';const im=document.createElement('img');im.src=f.file;im.alt=c.case_id+' '+f.offset+'초';a.append(im);const cap=document.createElement('figcaption');cap.textContent=f.offset===0?'기준 원본':(f.offset>0?'+':'')+f.offset+'초';fig.append(a,cap);box.append(fig);}const a=answers[c.case_id]||{};document.querySelectorAll('[data-field]').forEach(e=>{e.value=a[e.dataset.field]||'';e.disabled=!activeRater;});document.getElementById('note').value=a.note||'';document.getElementById('note').disabled=!activeRater;document.getElementById('prev').disabled=index===0;document.getElementById('next').disabled=index===cases.length-1;}
document.querySelectorAll('[data-field]').forEach(e=>e.addEventListener('change',save));document.getElementById('note').addEventListener('change',save);
rater.addEventListener('input',()=>{activeRater='';answers={};show();status.textContent='판정자 적용을 누르세요.';});
document.getElementById('apply').onclick=()=>{activeRater=rater.value.trim();answers={};if(activeRater)try{answers=JSON.parse(localStorage.getItem(key())||'{}');}catch(e){status.textContent='저장된 답안 읽기 실패';}show();status.textContent=activeRater?'판정자 적용됨: '+activeRater:'판정자 이름을 입력하세요.';};
document.getElementById('prev').onclick=()=>{save();index--;show();};document.getElementById('next').onclick=()=>{save();index++;show();};
document.getElementById('download').onclick=()=>{if(!save())return;const cell=v=>'"'+String(v??'').replaceAll('"','""')+'"';const rows=[fields,...cases.map(c=>fields.map(k=>(answers[c.case_id]||{case_id:c.case_id,rater:activeRater})[k]||''))];const blob=new Blob(['\\ufeff'+rows.map(r=>r.map(cell).join(',')).join('\\r\\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='temporal_observations_'+activeRater.replace(/[^a-zA-Z0-9가-힣_-]/g,'_')+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};show();
</script></html>'''


def prepare(root, out, seed=73):
    root, out = root.resolve(), out.resolve()
    if not out.is_relative_to(root.parent) or out.is_relative_to(root) or out.exists():
        raise ValueError("Use a new private sibling output folder, outside original data")
    pack = root / "label_pack"
    human = root / "human_review_50/labels_성진.csv"
    sheet_path, map_path = pack / "label_sheet_blind.csv", root / "private_frame_map.csv"
    paths = [human, sheet_path, map_path, root / "labels_claude.csv",
             root.parent / "ai_label_review_20261104/labels_codex.csv",
             root.parent / "road_visibility_20261104/run30/predictions.csv"]
    before = {str(p): fingerprint(p) for p in paths}
    sheet = pd.read_csv(sheet_path, dtype={"id": str})
    mapping = pd.read_csv(map_path, dtype={"id": str})
    selected = pd.read_csv(human, dtype={"id": str})
    rows = match_rows(sheet, mapping, selected)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    rows = rows.iloc[order].reset_index(drop=True)
    review = out / "review"
    review.mkdir(parents=True)
    archived = out / "source_snapshot"
    archived.mkdir()
    executed_source = archived / Path(__file__).name
    shutil.copyfile(__file__, executed_source)
    method = dict(status="running", seed=seed, rows=len(rows), offsets=OFFSETS, input_sha256=before,
                  source_sha256=fingerprint(executed_source), interpretation="Annotation reference only; not human vision measurement, new labels or model inputs")
    def write_method():
        (out / "method.json").write_text(json.dumps(method, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_method()
    cases, audit, private = [], [], []
    captures, video_info = {}, {}
    try:
        for num, r in enumerate(rows.itertuples(), 1):
            case_id = f"T{num:03d}"
            image_path = under(pack, r.file)
            vdir = under(root, str(r.video))
            videos = [p for p in vdir.glob("video.*") if p.suffix.lower() in [".mp4", ".webm", ".mkv"]]
            if len(videos) != 1:
                raise ValueError("Expected one original video per mapped sequence")
            video = videos[0]
            if r.video not in captures:
                cap = cv2.VideoCapture(str(video))
                if not cap.isOpened():
                    raise ValueError("Cannot open original video")
                fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if not np.isfinite([fps, count]).all() or fps <= 0 or count <= 0:
                    cap.release()
                    raise ValueError("Invalid video FPS/frame count")
                captures[r.video] = cap
                video_info[r.video] = dict(file=str(video.relative_to(root)), sha256=fingerprint(video),
                                          fps=fps, frame_count=int(count), duration_estimate_seconds=count/fps)
            cap = captures[r.video]
            source = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if source is None:
                raise ValueError("Cannot decode original labelled image")
            cdir = review / case_id
            cdir.mkdir()
            frames = []
            for offset in OFFSETS:
                requested = float(r.t)+offset
                if requested >= video_info[r.video]["duration_estimate_seconds"]:
                    raise ValueError("Neighbor time exceeds video duration; never repeat the last frame")
                image, timing = read_at(cap, requested)
                target = cdir / ("center.jpg" if offset == 0 else f"neighbor_{offset:+.0f}.jpg")
                if offset == 0:
                    if source.shape != image.shape:
                        raise ValueError("Video center does not match original image dimensions")
                    mae = float(np.abs(source.astype(float)-image.astype(float)).mean())
                    shutil.copyfile(image_path, target)
                else:
                    save_jpeg(target, image)
                audit.append(dict(case_id=case_id, id=r.id, video=r.video, offset=offset, **timing,
                                  output_sha256=fingerprint(target), center_decode_mae=mae if offset == 0 else None))
                frames.append(dict(offset=offset, file=f"{case_id}/{target.name}"))
            private.append(dict(case_id=case_id, id=r.id, video=r.video, t=float(r.t), original_sha256=fingerprint(image_path)))
            cases.append(dict(case_id=case_id, center_sha256=fingerprint(image_path), frames=frames))
            print(f"prepared {num}/{len(rows)}", flush=True)
        pd.DataFrame(audit).to_csv(out / "frame_audit.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(private).to_csv(out / "private_id_map.csv", index=False)
        (review / "review.html").write_text(render(cases), encoding="utf-8")
        with (review / "observations_blank.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({"case_id": c["case_id"]} for c in cases)
        if before != {str(p): fingerprint(p) for p in paths}:
            raise AssertionError("Original labels/predictions/source mapping changed")
        method.update(status="complete", videos=video_info, frames=len(audit),
                      center_decode_mae_max=max(a["center_decode_mae"] for a in audit if a["offset"] == 0),
                      limitations=["Future frames are annotation evidence only", "Whole videos must remain grouped", "No original labels or training inputs changed", "Original image watermarks remain", "Camera exposure and viewpoint are not calibrated", "Scene/reference existence is not driver vision validation"])
        write_method()
        print(f"complete: {review / 'review.html'}", flush=True)
    except BaseException as error:
        method.update(status="failed", error=repr(error))
        write_method()
        raise
    finally:
        for cap in captures.values():
            cap.release()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/new_cameras"))
    parser.add_argument("--out", type=Path, default=Path("data/temporal_visibility_20261104"))
    args = parser.parse_args()
    prepare(args.root, args.out)
