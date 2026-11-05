import importlib
from pathlib import Path

from PIL import Image
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'notebooks/experiments/sun_visible_probe'))
    return importlib.import_module('sd_supplement_labels')


def data(module, tmp_path):
    source = tmp_path / 'content'
    (source / 'content_candidates/images').mkdir(parents=True)
    rows, metadata = [], []
    for i in range(69):
        sid = f'sd_{i:016x}'
        image = source / 'content_candidates/images' / (sid+'.jpg')
        Image.new('RGB', (32, 18), (i, 100, 80)).save(image)
        rows.append(dict(case_id=f'D{i:04d}', stable_id=sid, clip_id=f'clip{i}', second='0',
                         image='content_candidates/images/'+image.name,
                         screen_reason='SECRET_SELECTION_NOTE', ai_reference_grade=''))
        metadata.append(dict(stable_id=sid, clip_id=f'clip{i}', group_id='20261103', sha256=module.base.sha(image)))
    module.base.write_csv(source / 'review_queue_69.csv', rows, list(rows[0]))
    module.base.dump(source / 'queue_rule_before_selection.json', dict(frozen=True))
    development, previous = tmp_path / 'development.csv', tmp_path / 'previous.csv'
    module.base.write_csv(development, metadata, list(metadata[0]))
    module.base.write_csv(previous, [dict(stable_id='previous_image')], ['stable_id'])
    return source, development, previous


def test_variable_last_batch_and_no_selection_notes_or_source_ids_in_prompts(module, tmp_path):
    source, development, previous = data(module, tmp_path)
    out, inputs = tmp_path / 'output', tmp_path / 'blind'
    manifest = module.prepare(source, development, previous, out, inputs)
    assert [len(p['ids']) for p in manifest['packets']] == [10]*6+[9]
    for packet in manifest['packets']:
        prompt = (Path(packet['cwd']) / 'prompt.txt').read_text(encoding='utf-8')
        assert 'SECRET_SELECTION_NOTE' not in prompt and 'sd_' not in prompt
        assert prompt.startswith(module.base.PROMPT)
        module.base.check_packet(packet)
    assert all(module.base.sha(p) == digest for p,digest in manifest['source_sha256'].items())
    with pytest.raises(ValueError, match='new output'):
        module.prepare(source, development, previous, out, inputs)


@pytest.mark.parametrize('bad', ['reserved', 'previous', 'nonblank'])
def test_forbidden_sources_rejected_before_output_creation(module, tmp_path, bad):
    source, development, previous = data(module, tmp_path)
    if bad == 'reserved':
        rows = module.read_csv(development)
        rows[0]['group_id'] = '20261105'
        module.base.write_csv(development, rows, list(rows[0]))
    elif bad == 'previous':
        module.base.write_csv(previous, [dict(stable_id='sd_0000000000000000')], ['stable_id'])
    else:
        rows = module.read_csv(source / 'review_queue_69.csv')
        rows[0]['ai_reference_grade'] = '2'
        module.base.write_csv(source / 'review_queue_69.csv', rows, list(rows[0]))
    with pytest.raises(ValueError):
        module.prepare(source, development, previous, tmp_path / 'output', tmp_path / 'blind')
    assert not (tmp_path / 'output').exists()


def test_known_quota_skips_requests_and_does_not_fabricate_failed_batches(module, monkeypatch, tmp_path):
    source, development, previous = data(module, tmp_path)
    out = tmp_path / 'output'
    module.prepare(source, development, previous, out, tmp_path / 'blind')
    evidence_file = tmp_path / 'retained_quota.json'
    module.base.dump(evidence_file, dict(error='real historical quota response'))
    module.base.dump(out / 'known_quota_evidence.json', {'claude': dict(evidence_file=str(evidence_file),
                      evidence_sha256=module.base.sha(evidence_file), reset_epoch=9999999999)})
    def no_request(*args, **kwargs):
        raise AssertionError('Known blocked provider must not be requested')
    monkeypatch.setattr(module.base, 'run_one', no_request)
    summary = module.run(out, ['claude'], 7)
    assert summary['claude']['completed'] == 0
    assert summary['claude']['failed_batches'] == []
    assert not (out / 'claude').exists()
    rows = module.read_csv(out / 'labels_claude.csv')
    assert len(rows) == 69 and {r['status'] for r in rows} == {'not_requested_known_quota'}
    assert summary['comparison']['incomplete'] == 69
    assert summary['comparison']['unanimous'] == 0
    assert module.quota_block(out, 'claude', now=9999999999) is None
