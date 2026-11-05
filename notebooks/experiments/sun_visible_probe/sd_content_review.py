"""Preserved SD development-only contact sheets and content-review candidates.

No predicted grades or brightness rankings select images. Thumbnail screening
is an auxiliary observation, not a severity label or a final test selection.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import random
import shutil

from PIL import Image, ImageDraw, ImageOps


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_source(root, relative):
    result = (root / relative).resolve()
    if not result.is_relative_to((root / 'images').resolve()):
        raise ValueError('Source image must remain inside root/images')
    return result


def prepare(root, candidates, previous, out):
    root, candidates, previous, out = [Path(p).resolve() for p in (root, candidates, previous, out)]
    if out.exists() or out.is_relative_to(root):
        raise ValueError('Use new private output outside original source')
    source = read_csv(candidates)
    old = read_csv(previous)
    if any(r['group_id'] == '20261105' for r in source):
        raise ValueError('Reserved date must not be screened')
    ids = {r['stable_id'] for r in source}
    old_ids = {r['stable_id'] for r in old}
    if len(ids) != len(source) or not old_ids <= ids:
        raise ValueError('Unique development IDs and a matching previous subset required')
    selected = [r for r in source if r['stable_id'] not in old_ids]
    if any(r.get('glare_label_final', '') for r in selected):
        raise ValueError('Expected unmodified blank source labels')
    random.Random(110550).shuffle(selected)
    out.mkdir(parents=True)
    (out / 'overview').mkdir()
    spec = dict(protocol='sd_development_content_screen_v1', seed=110550, source_root=str(root),
                source_sha256={str(p):sha(p) for p in (candidates, previous, Path(__file__))},
                excluded_previous_images=len(old_ids), development_images=len(source), new_images=len(selected),
                columns=5, rows=6, thumbnail=[320, 180],
                method='All remaining development candidates, randomized order, content-only blind IDs; no model scores/brightness ranks',
                screen='Flag plausible solar disc/bloom, forward-information-loss candidates and confounder/uncertain examples; no forced grade/class balance',
                limits='Thumbnail screening can miss subtle glare, cannot establish solar cause/grade2/human vision. Full-resolution follow-up required. Not independent of GPT-family annotation.',
                reserved_day_used=False, original_labels_changed=False)
    dump(out / 'specification_before_screen.json', spec)
    pairs, pages = [], []
    for start in range(0, len(selected), 30):
        page = start // 30 + 1
        canvas = Image.new('RGB', (1600, 1224), '#eeeeee')
        draw = ImageDraw.Draw(canvas)
        page_ids = []
        for index, row in enumerate(selected[start:start+30]):
            cid = f'D{start+index+1:04d}'
            original = safe_source(root, row['image'])
            if sha(original) != row['sha256']:
                raise ValueError('Original source hash changed')
            with Image.open(original) as image:
                tile = ImageOps.contain(image.convert('RGB'), (320, 180))
                x, y = (index % 5)*320, (index // 5)*204
                canvas.paste(tile, (x, y+24))
                draw.text((x+5, y+5), cid, fill='black')
            pairs.append(dict(case_id=cid, stable_id=row['stable_id'], image=row['image'],
                              group_id=row['group_id'], clip_id=row['clip_id'], second=row['second'],
                              sha256=row['sha256'], page=page))
            page_ids.append(cid)
        path = out / 'overview' / f'sheet_{page:03d}.jpg'
        canvas.save(path, quality=92)
        pages.append(dict(page=page, case_ids=page_ids, contact_sheet_sha256=sha(path)))
    write_csv(out / 'private_mapping.csv', pairs, list(pairs[0]))
    dump(out / 'pages.json', pages)
    shutil.copyfile(__file__, out / 'source_at_prepare.py')
    dump(out / 'report.json', dict(stage='ready_for_content_screen', new_images=len(pairs), pages=len(pages),
                                   source_unchanged=all(sha(p)==h for p,h in spec['source_sha256'].items()),
                                   reserved_day_used=False, severity_labels_created=0))
    return dict(new_images=len(pairs), pages=len(pages))


def finalize(out, observations):
    out = Path(out).resolve()
    spec = json.loads((out / 'specification_before_screen.json').read_text(encoding='utf-8'))
    pairs = read_csv(out / 'private_mapping.csv')
    pages = json.loads((out / 'pages.json').read_text(encoding='utf-8'))
    reviews = json.loads(Path(observations).read_text(encoding='utf-8'))
    by_id = {r['case_id']: r for r in pairs}
    if {r['page'] for r in reviews} != {p['page'] for p in pages} or len(reviews) != len(pages):
        raise ValueError('Exactly one observation record for every viewed page required')
    candidates = []
    for review in reviews:
        page = next(p for p in pages if p['page'] == review['page'])
        if sha(out / 'overview' / f"sheet_{page['page']:03d}.jpg") != page['contact_sheet_sha256']:
            raise ValueError('Contact sheet changed')
        if not review.get('viewed') or not review.get('note'):
            raise ValueError('Missing page review evidence')
        for flagged in review['flagged']:
            if flagged['case_id'] not in page['case_ids'] or not flagged['reason']:
                raise ValueError('Invalid content flag or case/page mismatch')
            candidates.append(dict(**by_id[flagged['case_id']], screen_reason=flagged['reason'],
                                   provenance='codex_assistant_thumbnail_screen_not_severity_truth'))
    if len({r['case_id'] for r in candidates}) != len(candidates):
        raise ValueError('Duplicate content candidate')
    if any(sha(p)!=h for p,h in spec['source_sha256'].items()):
        raise ValueError('Frozen source changed')
    raw = Path(spec['source_root'])
    image_dir = out / 'content_candidates' / 'images'
    image_dir.mkdir(parents=True, exist_ok=False)
    for row in candidates:
        original = safe_source(raw, row['image'])
        if sha(original) != row['sha256']:
            raise ValueError('Source image changed')
        destination = image_dir / (row['stable_id'] + '.jpg')
        shutil.copyfile(original, destination)
        if sha(destination) != row['sha256']:
            raise ValueError('Candidate copy must preserve original bytes')
    fields = ['case_id', 'stable_id', 'image', 'group_id', 'clip_id', 'second', 'sha256', 'page', 'screen_reason', 'provenance']
    write_csv(out / 'content_candidates.csv', candidates, fields)
    cards = []
    for row in candidates:
        cards.append('<article><h2>' + row['case_id'] + '</h2><img loading="lazy" src="images/' + row['stable_id'] + '.jpg"><p>' + html.escape(row['screen_reason']) + '</p></article>')
    (image_dir.parent / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>SD content candidates</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto}img{width:100%}article{padding:12px;border-bottom:1px solid #aaa}</style><h1>내용 검토 후보 — 눈부심 확정 라벨 아님</h1><p>기존300장/보류날짜 제외. 썸네일의 태양/번짐·식별저하 의심/혼동사례. 원본해상도 확인 후 판단.</p>' + ''.join(cards), encoding='utf-8')
    dump(out / 'report.json', dict(stage='thumbnail_screen_complete', development_images=spec['development_images'],
         previous_images_excluded=spec['excluded_previous_images'], newly_screened=len(pairs), reviewed_pages=len(reviews),
         content_candidates=len(candidates), candidate_clips=len({r['clip_id'] for r in candidates}),
         no_flag_means='No salient candidate identified at thumbnail resolution, not a negative label',
         source_unchanged=True, reserved_day_used=False, severity_labels_created=0,
         independent_third_rater=False, model_training_runs=0))
    return json.loads((out / 'report.json').read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'finalize'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--observations', type=Path)
    args = parser.parse_args()
    result = prepare(args.root, args.candidates, args.previous, args.out) if args.action == 'prepare' else finalize(args.out, args.observations)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
