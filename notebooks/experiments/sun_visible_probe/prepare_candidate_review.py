"""Read saved OOF/held-camera results into a private, photo-backed review page.

No fitting or model/threshold selection. Full-data candidate predictions are
never substituted for held-out scores. Original photos are linked locally.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "glare_video_source"))

import numpy as np
import pandas as pd

from nunchi_ml.glare.evaluation import evaluate_grades
import g_visibility_probe as C

METHODS = {
    "B_geometry_aug": "사진+G · B 보강 후보",
    "source_conditional": "사진+G · 원자료 조건부",
    "source_rule_tuned": "원자료 · 튜닝 규칙",
    "aug_rule_tuned": "추가자료 · 튜닝 규칙",
}
SCORES = {
    "B_geometry_aug": ("source_a", "geometry_b"),
    "source_conditional": ("source_a", "source_b"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_saved(predictions, probabilities, originals):
    """Require exact test identities and targets, never repair coverage by join."""
    if predictions.duplicated(["test", "repeat", "method", "id"]).any():
        raise ValueError("Duplicate saved predictions")
    if probabilities.duplicated(["test", "repeat", "id"]).any():
        raise ValueError("Duplicate saved probabilities")
    if not originals.id.is_unique:
        raise ValueError("Unique frozen original IDs required")
    rows = []
    for (regime, repeat, method), part in predictions.groupby(["test", "repeat", "method"], sort=False):
        if method not in METHODS:
            continue
        expected = originals if regime == "group_cv320" else originals[originals.camera.eq(regime.removeprefix("held_camera:"))]
        if set(part.id) != set(expected.id) or len(part) != len(expected):
            raise ValueError("Saved prediction coverage differs from frozen capture test")
        part = part.set_index("id").loc[expected.id].reset_index()
        if not np.array_equal(part.label, expected.label) or not np.array_equal(part.group, expected.group):
            raise ValueError("Frozen reference grades/captures changed")
        joint, b = None, None
        if method in SCORES:
            score = probabilities[probabilities.test.eq(regime) & probabilities["repeat"].eq(repeat)]
            if set(score.id) != set(expected.id) or len(score) != len(expected):
                raise ValueError("Saved probability coverage differs from prediction coverage")
            score = score.set_index("id").loc[expected.id]
            a, b = (score[c].to_numpy(float) for c in SCORES[method])
            if not np.array_equal(np.where(a >= .5, np.where(b >= .5, 2, 1), 0), part.pred):
                raise ValueError("Binary scores do not reproduce saved .5/.5 decisions")
            joint = np.column_stack([1 - a, a * (1 - b), a * b])
        metric = evaluate_grades(part.label.to_numpy(), part.pred.to_numpy(), joint, b)
        rows.append(dict(test=regime, repeat=int(repeat), method=method, **metric))
    return pd.DataFrame(rows)


def page(payload):
    # Escape script terminators even if input labels/paths are later untrusted.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return TEMPLATE.replace("__PAYLOAD__", encoded)


def run(data, out, *, selected_prototype=None):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(data):
        raise ValueError("Use a new private output under data")
    if selected_prototype not in (None, "B_geometry_aug", "source_conditional"):
        raise ValueError("Explicitly confirmed prototype must be an existing photo+G candidate")
    prior = data / "conditional_augmentation_20261105"
    inputs = [prior / n for n in ("report.json", "predictions.csv", "binary_probabilities.csv", "summary.csv", "paired_capture_intervals.csv")]
    snapshot = data / "road_visibility_20261104/run30/training_features.csv"
    hashes = {str(p): sha(p) for p in [*inputs, snapshot]}
    provenance = json.loads(inputs[0].read_text(encoding="utf-8"))
    if provenance["status"] != "complete" or any(sha(p) != v for p, v in provenance["input_sha256"].items()):
        raise ValueError("Existing experiment provenance changed")
    full = pd.read_csv(snapshot, dtype={"id": str})
    originals = full[full.camera.ne("AI Hub")].reset_index(drop=True)
    if len(originals) != 320 or originals.group.nunique() != 24:
        raise ValueError("Frozen original320/24capture required")
    predictions = pd.read_csv(inputs[1], dtype={"id": str})
    probabilities = pd.read_csv(inputs[2], dtype={"id": str})
    metrics = score_saved(predictions, probabilities, originals)
    if len(metrics) != 132:
        raise ValueError("Expected four methods x (30 CV repeats + three camera tests)")
    old = pd.read_csv(inputs[3]).set_index(["test", "method"])
    summary = []
    for (regime, method), scores in metrics.groupby(["test", "method"], sort=False):
        row = dict(test=regime, method=method, name=METHODS[method], repeats=len(scores))
        for name in ["macro_f1", "weighted_f1", "accuracy", "recall_2", "count_0_to_2", "count_2_to_0", "f1_1"]:
            value = float(scores[name].mean())
            if not np.isclose(value, old.loc[(regime, method), name], atol=1e-12, rtol=0):
                raise ValueError("Common evaluation differs from preserved summary")
            row[name] = value
            row[name + "_std"] = float(scores[name].std(ddof=0))
        for name in ["macro_ovr_auc", "macro_average_precision", "multiclass_log_loss", "multiclass_brier"]:
            row[name] = None if scores[name].isna().all() else float(scores[name].mean())
        summary.append(row)
    paths = C.image_paths(originals, data)
    lookup = {}
    for row, path in zip(originals.itertuples(), paths):
        if sha(path) != row.image_sha256:
            raise ValueError("Original photo bytes changed")
        lookup[row.id] = dict(id=row.id, image=os.path.relpath(path, out).replace(os.sep, "/"),
                             label=int(row.label), camera=row.camera, group=row.group,
                             captured_at=row.captured_at_kst, sun_alt=float(row.sun_alt),
                             gap=float(row.gap), cloud=float(row.cloud), sunshine=float(row.sunshine))
    case_rows = predictions[predictions.method.isin(METHODS) & predictions["repeat"].isin([0, -1])]
    cases = {}
    for (regime, identity), rows in case_rows.groupby(["test", "id"], sort=False):
        item = dict(test=regime, **lookup[identity], predictions={})
        score = probabilities[probabilities.test.eq(regime) & probabilities.id.eq(identity) & probabilities["repeat"].isin([0, -1])]
        if len(score) != 1:
            raise ValueError("Each displayed case requires one saved held-out score row")
        for row in rows.itertuples():
            prediction = dict(level=int(row.pred), joint_scores=None, a=None, b=None)
            if row.method in SCORES:
                a, b = (float(score.iloc[0][c]) for c in SCORES[row.method])
                prediction.update(joint_scores=[1 - a, a * (1 - b), a * b], a=a, b=b)
            item["predictions"][row.method] = prediction
        if set(item["predictions"]) != set(METHODS):
            raise ValueError("Every case needs all four preserved comparators")
        cases[(regime, identity)] = item
    intervals = pd.read_csv(inputs[4])
    intervals = intervals[intervals.method.eq("B_geometry_aug")].to_dict("records")
    payload = dict(title="모델② · 후보 검토", status="review_only", production_model_replaced=False,
                   selected_prototype_model=selected_prototype,
                   default_candidate="B_geometry_aug", methods=METHODS, summary=summary,
                   paired_intervals=intervals, cases=list(cases.values()),
                   recommendation="시제품 비교 기본값은 기존 B_geometry_aug 후보. 최종 채택·운영교체는 미확정.",
                   policy="Same frozen labels, captures, folds and .5/.5 decisions; saved predictions only",
                   development_limit="원320/24촬영·30회는 중복 개발평가. 사진별 결과는 OOF 첫 회차 또는 카메라 제외 예측.")
    if any(sha(p) != v for p, v in hashes.items()):
        raise ValueError("Input changed during saved-result review")
    out.mkdir(parents=True)
    (out / "index.html").write_text(page(payload), encoding="utf-8")
    (out / "review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    metrics.to_csv(out / "saved_metrics.csv", index=False, encoding="utf-8-sig")
    result = dict(status="complete", fitting_runs=0, thresholds_changed=False, grades_changed=False,
                  preserved_metric_rows=132, original_photos_checked=320, displayed_heldout_cases=len(cases),
                  production_model_replaced=False, selected_prototype_model=selected_prototype,
                  prototype_selection_confirmed=selected_prototype is not None,
                  review_source_sha256=sha(__file__), input_sha256=hashes,
                  original_image_sha256={str(p): sha(p) for p in paths})
    (out / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("sha256")}, ensure_ascii=False), flush=True)
    return result


TEMPLATE = r'''<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>눈치 · 모델② 후보 검토</title>
<style>
:root{color-scheme:light;font-family:"Malgun Gothic",system-ui,sans-serif;color:#202827;background:#f4f6f3}*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;padding:32px 24px}header{display:flex;justify-content:space-between;align-items:center;gap:16px}.eyebrow{color:#527360;font-size:12px;font-weight:700;letter-spacing:.12em}h1{font-size:32px;margin:9px 0 12px;letter-spacing:-1px}h2{font-size:19px;margin:0 0 16px}p{line-height:1.75;margin:8px 0}.tag{border:1px solid #bbd1c0;color:#31543d;border-radius:20px;padding:8px 14px;white-space:nowrap;font-size:12px}.notice{background:#fff6dd;border-left:4px solid #d5a334;padding:16px 20px;margin:24px 0;font-size:13px}.panel{background:white;border:1px solid #e0e7e0;border-radius:16px;padding:24px;margin:20px 0;box-shadow:0 2px 12px #243c2c05}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.stat{padding:18px;background:#eaf1e9;border-radius:12px}.stat small{display:block;color:#607565;font-size:12px;margin-bottom:10px}.stat b{font-size:27px;font-variant-numeric:tabular-nums}.stat em{font-size:12px;font-style:normal;display:block;margin-top:7px;color:#506454}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{text-align:right;padding:14px 12px;border-bottom:1px solid #edf0eb;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}th{background:#f7f9f6;color:#536554}.highlight{background:#f0f6ed}.notes{color:#627061;font-size:12px}.toolbar{display:flex;flex-wrap:wrap;gap:14px;align-items:end;margin:18px 0}label{font-size:12px;display:grid;gap:7px;color:#586658}select,button{border:1px solid #ced8cc;border-radius:8px;background:#fff;padding:10px 12px;font:inherit;color:#273c2c}button{cursor:pointer;background:#315d40;color:white}.gallery{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}.case{border:1px solid #e2e7df;border-radius:12px;overflow:hidden;background:white}.case img{width:100%;height:248px;object-fit:contain;background:#18201d;display:block}.casebody{padding:16px}.casehead{display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:13px}.badge{padding:5px 10px;border-radius:7px;font-size:12px;background:#e5f0e3;color:#365d3c}.badge.error{background:#fbe5df;color:#943f2a}.meta{font-size:12px;color:#687168;line-height:1.8;margin-top:10px}.bar{height:6px;background:#e9eee7;border-radius:9px;overflow:hidden;margin-top:7px}.bar i{display:block;height:100%;background:#729c69}.scores{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:13px;font-size:11px}.empty{text-align:center;padding:30px;color:#64765f}.limit{max-width:980px;font-size:13px}.foot{margin:24px 0;color:#627061;font-size:12px}@media(max-width:700px){main{padding:22px 14px}h1{font-size:25px}header{align-items:start;flex-direction:column}.stats{grid-template-columns:repeat(2,1fr)}.panel{padding:18px}.gallery{grid-template-columns:1fr}.case img{height:210px}th,td{padding:11px 9px}}
</style><main>
<header><div><div class="eyebrow">NUNCHI / MODEL REVIEW</div><h1>눈부심 모델, 같은 조건에서 보기</h1><p class="notes">기존 사진+G 후보 · 저장된 검증 결과와 원본 사진을 함께 확인합니다.</p></div><span class="tag" id="selection-status">검토용 · 기준 선택 전</span></header>
<div class="notice"><b>구현한 모델과 확장 구상을 구분합니다.</b><br>아래는 보유 사진의 개발평가 결과입니다. 도시 CCTV·미래 경로·사고 감소 성능은 검증하지 않았습니다. 시뮬레이터의 체험값을 모델 성적으로 사용하지 않습니다.</div>
<div class="stats" id="stats"></div>
<section class="panel"><h2>후보와 강한 규칙 비교</h2><p class="notes">원320장·24촬영, 같은 5겹 × 30회 평균. 낮을수록 좋은 오류 건수는 회차당 평균입니다. 각 회차는 독립된 시험이 아닙니다.</p><div class="scroll"><table id="comparison"></table></div><div class="notes" id="intervals"></div></section>
<section class="panel"><h2>카메라가 바뀌면</h2><p class="notes">해당 카메라 전체를 학습에서 제외한 결과. 사진+G 두 후보를 나란히 봅니다.</p><div class="scroll"><table id="cameras"></table></div></section>
<section class="panel"><h2>실제 사진의 검증 예측</h2><p class="notes">첫 OOF 회차 또는 카메라 제외 예측만 표시합니다. 최종 전체학습 모델의 사진 암기 성적이 아닙니다. 전체 사례를 조회할 수 있고 오답도 숨기지 않습니다.</p><div class="toolbar"><label>검증 조건<select id="regime"></select></label><label>비교할 방법<select id="method"></select></label><label>정답 등급<select id="grade"><option value="all">전체</option><option value="0">정상</option><option value="1">눈부심</option><option value="2">심한 눈부심</option></select></label><label>사례<select id="errors"><option value="all">전체</option><option value="wrong">오답</option><option value="miss">심함 → 정상 누락</option><option value="false">정상 → 심함 오경보</option></select></label><button id="more">다음 사례</button></div><p class="notes" id="casecount"></p><div class="gallery" id="gallery"></div></section>
<section class="panel limit"><h2>발표용 기준과 남은 판단</h2><p><b>비교 화면의 기본값은 기존 사진+G 보강 후보입니다.</b> F1 .6396과 정상→심함 오경보 감소를 관측했지만, 원자료 조건부보다 심함 재현율이 낮습니다. 최종 모델 채택이나 안전성 인증을 뜻하지 않습니다.</p><p>보강 후보와 원자료 조건부의 F1 차이 구간은 −.0080 ~ +.0391로 0을 포함합니다. 추가자료 접근이 같은 튜닝 규칙과의 차이도 −.0037 ~ +.1372로 우위가 확실하지 않습니다. 오류 부담과 심함 누락 중 무엇을 우선할지는 최종 판단이 필요합니다.</p><p>독립 사람시험·다른 촬영의 심함 표본이 부족합니다. 원320의 심함73장 중71장이 한 촬영에 집중돼 있어 새 환경의 안정성을 보장할 수 없습니다. CCTV 실연결은 확장 구상으로 남깁니다.</p></section><div class="foot">이 페이지와 원본 사진은 로컬 검토용입니다. 외부 공유 폴더나 공개 저장소에 업로드하지 않습니다.</div></main>
<script>const D=__PAYLOAD__;
const $=id=>document.getElementById(id), grades=['정상','눈부심','심한 눈부심'], fmt=(n,k=4)=>n==null?'미정의':Number(n).toFixed(k);
if(D.selected_prototype_model)$('selection-status').textContent='시제품 기준 확정 · '+D.methods[D.selected_prototype_model];
function cell(row,value,tag='td'){const e=document.createElement(tag);e.textContent=value;row.append(e)}
function table(id,heads,rows){const el=$(id);el.replaceChildren();let tr=document.createElement('tr');heads.forEach(v=>cell(tr,v,'th'));el.append(tr);for(const values of rows){tr=document.createElement('tr');values.forEach(v=>cell(tr,v));el.append(tr)}}
const cv=D.summary.filter(s=>s.test==='group_cv320'), primary=cv.find(s=>s.method===D.default_candidate);
for(const [label,value,note] of [['보강 후보 Macro F1',fmt(primary.macro_f1),'반복 촬영 CV 평균'],['심함 재현율',fmt(primary.recall_2*100,1)+'%','놓친 심함도 함께 확인'],['검증 사진','320장','24촬영 · 재사용 개발평가'],['추가 학습 / 경계 변경','0회','저장 결과만 재평가']]){const e=document.createElement('div');e.className='stat';const sm=document.createElement('small'),b=document.createElement('b'),em=document.createElement('em');sm.textContent=label;b.textContent=value;em.textContent=note;e.append(sm,b,em);$('stats').append(e)}
table('comparison',['방법','Macro F1','정확도','중간 F1','심함 재현율','정상→심함','심함→정상'],cv.map(s=>[s.name,fmt(s.macro_f1),fmt(s.accuracy),fmt(s.f1_1),fmt(s.recall_2),fmt(s.count_0_to_2,2),fmt(s.count_2_to_0,2)]));
$('intervals').textContent='F1 차이의 고정 OOF 촬영군집 구간(모두 0 포함): '+D.paired_intervals.map(s=>D.methods[s.baseline]+' 대비 '+fmt(s.low)+' ~ '+fmt(s.high)).join(' / ');
table('cameras',['제외한 카메라','방법','Macro F1','심함 재현율','심함→정상'],D.summary.filter(s=>s.test!=='group_cv320'&&['B_geometry_aug','source_conditional'].includes(s.method)).map(s=>[s.test.replace('held_camera:',''),s.name,fmt(s.macro_f1),fmt(s.recall_2),fmt(s.count_2_to_0,0)]));
for(const r of [...new Set(D.cases.map(c=>c.test))]){const e=document.createElement('option');e.value=r;e.textContent=r==='group_cv320'?'촬영 OOF · 첫 회차':r.replace('held_camera:','카메라 제외 · ');$('regime').append(e)}
for(const [key,name] of Object.entries(D.methods)){const e=document.createElement('option');e.value=key;e.textContent=name;$('method').append(e)}$('method').value=D.default_candidate;
let offset=0;function render(){const method=$('method').value,filter=$('errors').value,grade=$('grade').value;const cases=D.cases.filter(c=>c.test===$('regime').value&&(grade==='all'||c.label===Number(grade))).filter(c=>{const p=c.predictions[method].level;return filter==='all'||filter==='wrong'&&p!==c.label||filter==='miss'&&c.label===2&&p===0||filter==='false'&&c.label===0&&p===2});if(offset>=cases.length)offset=0;$('casecount').textContent=cases.length+'건 중 '+(cases.length?offset+1:0)+'–'+Math.min(offset+6,cases.length)+'건 · 정렬: 저장된 검증 순서';$('gallery').replaceChildren();for(const c of cases.slice(offset,offset+6)){const p=c.predictions[method],card=document.createElement('article');card.className='case';const image=document.createElement('img');image.src=c.image;image.alt=c.camera+' 검증 사진 '+c.id;image.loading='lazy';image.onerror=()=>{image.alt='원본 사진을 읽지 못했습니다 · '+c.id};const body=document.createElement('div');body.className='casebody';const head=document.createElement('div');head.className='casehead';const name=document.createElement('b');name.textContent=c.camera;const badge=document.createElement('span');badge.className='badge'+(p.level!==c.label?' error':'');badge.textContent='정답 '+grades[c.label]+' / 예측 '+grades[p.level];head.append(name,badge);const meta=document.createElement('div');meta.className='meta';meta.textContent=c.captured_at+' · 해 높이 '+fmt(c.sun_alt,1)+'° · 방향차 '+fmt(c.gap,1)+'° · 운량 '+c.cloud+'\nID '+c.id;body.append(head,meta);const scores=document.createElement('div');scores.className='scores';if(p.joint_scores){p.joint_scores.forEach((n,i)=>{const d=document.createElement('div');d.textContent=grades[i]+' '+fmt(n*100,1)+'%';const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('i');fill.style.width=(n*100)+'%';bar.append(fill);d.append(bar);scores.append(d)});const line=document.createElement('div');line.className='notes';line.textContent='모델 점수(보정 전) · A '+fmt(p.a,3)+' / B '+fmt(p.b,3)+' · .5/.5 판정';body.append(scores,line)}else{const line=document.createElement('div');line.className='notes';line.textContent='규칙의 등급 판정 · 확률 점수 없음';body.append(line)}card.append(image,body);$('gallery').append(card)}if(!cases.length){const e=document.createElement('p');e.className='empty';e.textContent='해당 조건의 사례가 없습니다.';$('gallery').append(e)}}
for(const id of ['regime','method','grade','errors'])$(id).addEventListener('change',()=>{offset=0;render()});$('more').addEventListener('click',()=>{offset+=6;render()});render();
</script></html>'''


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--selected-prototype", choices=["B_geometry_aug", "source_conditional"],
                   help="Only record a prototype choice explicitly confirmed by the user; never infer it from scores")
    a = p.parse_args()
    run(a.data, a.out, selected_prototype=a.selected_prototype)
