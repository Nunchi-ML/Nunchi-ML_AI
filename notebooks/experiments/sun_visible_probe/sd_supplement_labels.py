"""New blinded 69-case development supplement; preserves the frozen 300-case run.

Reuses the identical rubric/schema and vendor transports, with explicit inherited
quota blocks. Content-selection notes and previous answers never enter prompts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import threading
import time

import sd_three_ai_labels as base


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def prepare(source, development, previous, out, inputs):
    source, development, previous, out, inputs = [Path(p).resolve() for p in (source, development, previous, out, inputs)]
    if out.exists() or inputs.exists():
        raise ValueError('Use new output/input directories')
    if (out.is_relative_to(source) or inputs.is_relative_to(source)
            or out.is_relative_to(inputs) or inputs.is_relative_to(out)):
        raise ValueError('Source, outputs and blinded inputs must be separate')
    queue_file = source / 'review_queue_69.csv'
    rows = read_csv(queue_file)
    lookup = {r['stable_id']: r for r in read_csv(development)}
    old_ids = {r['stable_id'] for r in read_csv(previous)}
    if len(rows) != 69 or len({r['stable_id'] for r in rows}) != 69 or len({r['clip_id'] for r in rows}) != 69:
        raise ValueError('Expected 69 distinct development images and clips')
    originals = {str(p): base.sha(p) for p in (queue_file, development, previous,
                 source / 'queue_rule_before_selection.json', Path(base.__file__), Path(__file__))}
    for row in rows:
        if row['stable_id'] in old_ids or row['stable_id'] not in lookup:
            raise ValueError('Supplement must exclude previous images and stay in development')
        meta = lookup[row['stable_id']]
        if meta['group_id'] == '20261105' or meta['clip_id'] != row['clip_id'] or row['ai_reference_grade']:
            raise ValueError('Reserved date, mismatched clip or nonblank grade rejected')
        image = (source / row['image']).resolve()
        if not image.is_relative_to(source / 'content_candidates/images') or base.sha(image) != meta['sha256']:
            raise ValueError('Original path/hash mismatch')
        row['_original'] = str(image)
        row['_sha256'] = meta['sha256']
        originals[str(image)] = meta['sha256']
    random.Random(110569).shuffle(rows)
    out.mkdir(parents=True)
    inputs.mkdir(parents=True)
    packets, pairs = [], []
    for start in range(0, 69, 10):
        batch = inputs / f'batch_{start//10+1:02d}'
        batch.mkdir()
        ids, images = [], {}
        for index, row in enumerate(rows[start:start+10], start+1):
            cid = f'R{index:03d}'
            target = batch / (cid + '.jpg')
            shutil.copyfile(row['_original'], target)
            if base.sha(target) != row['_sha256']:
                raise ValueError('Byte-identical copies required')
            ids.append(cid)
            images[str(target)] = row['_sha256']
            pairs.append(dict(case_id=cid, stable_id=row['stable_id'], original=row['_original'], sha256=row['_sha256']))
        prompt = base.PROMPT + '\nCASES: ' + ', '.join(ids) + '\nEach attachment filename identifies its case.\n'
        (batch / 'prompt.txt').write_text(prompt, encoding='utf-8')
        base.dump(batch / 'schema.json', base.schema(ids))
        packets.append(dict(name=batch.name, cwd=str(batch), ids=ids, images=images,
                            prompt_sha256=base.sha(batch / 'prompt.txt'), schema_sha256=base.sha(batch / 'schema.json')))
    manifest = dict(protocol='sd_supplement69_ai_reference_v1', source_sha256=originals, packets=packets,
                    seed=110569, batch_size=10, image_count=69, providers=list(base.PROVIDERS),
                    rubric_sha256=hashlib.sha256(base.PROMPT.encode()).hexdigest(),
                    base_runner_sha256=base.sha(base.__file__), source_code_sha256=base.sha(__file__),
                    source_note='Content-enriched development supplement; shares clips with initial300, not representative prevalence or independent test',
                    blinded='Only opaque IDs, identical rubric/schema and original JPEG bytes; no selection notes, source names, old grades or model predictions',
                    human_ground_truth=False, paid_credit_fallback=False, reserved_day_used=False)
    base.dump(out / 'manifest.json', manifest)
    base.dump(out / 'private_pairs.json', pairs)
    shutil.copyfile(__file__, out / 'source_at_prepare.py')
    shutil.copyfile(base.__file__, out / 'base_runner_at_prepare.py')
    return manifest


def inherit_quota_evidence(old, out):
    """No network requests. Record known failures as availability, not new batches."""
    evidence = {}
    old, out = Path(old).resolve(), Path(out).resolve()
    path = old / 'claude/batch_16/status.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('provider_error') != 'rate_limit' or value['valid']:
        raise ValueError('Expected retained Claude quota evidence')
    evidence['claude'] = dict(reason='retained_weekly_quota_rejection', evidence_file=str(path),
                              evidence_sha256=base.sha(path), reset_epoch=1793959200,
                              reset_source='provider resetsAt epoch', no_new_request=True)
    path = old / 'gemini/batch_12/status.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    transcript = old / 'gemini/batch_12/stdout.jsonl'
    if value.get('provider_error') != 'quota_reached' or value['valid'] or '144h36m' not in value['provider_message']:
        raise ValueError('Expected retained Gemini quota evidence')
    evidence['gemini'] = dict(reason='retained_individual_quota_rejection', evidence_file=str(path),
                              evidence_sha256=base.sha(path), transcript_sha256=base.sha(transcript),
                              reset_epoch=transcript.stat().st_mtime + 144*3600 + 36*60 + 60,
                              reset_source='approximate countdown from unchanged response transcript timestamp plus60s; not provider absolute epoch',
                              no_new_request=True)
    base.dump(out / 'known_quota_evidence.json', evidence)
    return evidence


def quota_block(out, provider, now=None):
    path = Path(out) / 'known_quota_evidence.json'
    evidence = json.loads(path.read_text(encoding='utf-8')).get(provider) if path.exists() else None
    if evidence:
        if base.sha(evidence['evidence_file']) != evidence['evidence_sha256']:
            raise ValueError('Quota evidence changed; inspect before resuming')
        if (time.time() if now is None else now) < evidence['reset_epoch']:
            return evidence
    return None


def export(out):
    out = Path(out).resolve()
    summary = base.export(out)
    for provider in base.PROVIDERS:
        block = quota_block(out, provider)
        if block:
            path = out / f'labels_{provider}.csv'
            rows = read_csv(path)
            for row in rows:
                if row['status'] == 'pending_or_failed':
                    row['status'] = 'not_requested_known_quota'
            base.write_csv(path, rows, list(rows[0]))
        summary[provider]['known_quota_blocked_before_request'] = bool(block)
    summary['supplement'] = 'Content-enriched 69-image development set; keep separate from initial300 and natural prevalence. No new human truth/F1.'
    base.dump(out / 'summary.json', summary)
    return summary


def run(out, providers, batches, workers=1, retry_failed=False):
    out = Path(out).resolve()
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    if base.sha(base.__file__) != manifest['base_runner_sha256']:
        raise ValueError('Frozen transport/rubric changed')
    journal = out / 'runner_history'
    journal.mkdir(exist_ok=True)
    stamp = str(time.time_ns())
    shutil.copyfile(__file__, journal / (stamp+'.py'))
    base.dump(journal / (stamp+'.json'), dict(providers=providers, batches=batches, workers=workers,
              retry_failed=retry_failed, recorded_utc=datetime.now(timezone.utc).isoformat()))
    for provider in providers:
        block = quota_block(out, provider)
        if block:
            print(json.dumps(dict(provider=provider, status='not_requested_known_quota', reset_epoch=block['reset_epoch'])), flush=True)
            continue
        stop = threading.Event()
        def worker(offset):
            for packet in manifest['packets'][:batches][offset::workers]:
                if stop.is_set():
                    break
                if not base.run_one(provider, packet, out, retry_failed)['valid']:
                    stop.set()
                    break
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, i) for i in range(workers)]
            for future in futures:
                future.result()
    return export(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'export'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--development', type=Path)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--quota-source', type=Path)
    parser.add_argument('--providers', nargs='+', choices=base.PROVIDERS, default=list(base.PROVIDERS))
    parser.add_argument('--batches', type=int, choices=range(1, 8), default=1)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=1)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        manifest = prepare(args.source, args.development, args.previous, args.out, args.inputs)
        inherit_quota_evidence(args.quota_source, args.out)
        export(args.out)
        print(json.dumps(dict(images=69, batches=[len(p['ids']) for p in manifest['packets']])))
    elif args.action == 'run':
        print(json.dumps(run(args.out, args.providers, args.batches, args.workers, args.retry_failed), ensure_ascii=True))
    else:
        print(json.dumps(export(args.out), ensure_ascii=True))


if __name__ == '__main__':
    main()
