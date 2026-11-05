import importlib
import csv
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'notebooks/experiments/sun_visible_probe'))
    return importlib.import_module('sd_three_ai_labels')


def item(cid='S001'):
    return dict(case_id=cid, pixels_viewed=True, scene='forward_or_aligned_road',
                sun_evidence='none', grade='0', visibility='clear', cause='none',
                evidence='Road boundaries visible; no solar bloom.', uncertainty='Distant details uncertain.')


def test_reserved_date_rejected_before_creating_outputs(module, tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    path = source / 'labels_blank.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        keys = ['id', 'image', 'clip_id', 'group_id', 'glare_label_rater1', 'glare_label_rater2', 'glare_label_final']
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(dict(id=str(i), image=f'images/{i}.jpg', clip_id=str(i), group_id='20261105',
                              glare_label_rater1='', glare_label_rater2='', glare_label_final='') for i in range(300))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Reserved day'):
        module.prepare(source, tmp_path / 'outputs', tmp_path / 'inputs')
    assert not (tmp_path / 'outputs').exists()
    assert not (tmp_path / 'inputs').exists()
    assert path.read_bytes() == before


def test_rejects_duplicate_and_missing_images(module):
    with pytest.raises(ValueError, match='case IDs'):
        module.validate({'items': [item(), item()]}, ['S001', 'S002'])
    with pytest.raises(ValueError, match='case IDs'):
        module.validate({'items': [item()]}, ['S001', 'S002'])


def test_unseen_or_wrong_view_cannot_receive_forward_grade(module):
    row = item()
    row['pixels_viewed'] = False
    with pytest.raises(ValueError, match='Out-of-scope'):
        module.validate({'items': [row]}, ['S001'])
    row.update(pixels_viewed=True, scene='rear')
    with pytest.raises(ValueError, match='Out-of-scope'):
        module.validate({'items': [row]}, ['S001'])
    row['grade'] = 'abstain'
    assert module.validate({'items': [row]}, ['S001']) == [row]


def test_changed_attachment_rejected_before_request(module, tmp_path):
    (tmp_path / 'prompt.txt').write_text('fixed', encoding='utf-8')
    (tmp_path / 'schema.json').write_text('{}', encoding='utf-8')
    image = tmp_path / 'S001.jpg'
    image.write_bytes(b'original bytes')
    packet = dict(cwd=str(tmp_path), prompt_sha256=module.sha(tmp_path / 'prompt.txt'),
                  schema_sha256=module.sha(tmp_path / 'schema.json'), images={str(image): module.sha(image)})
    module.check_packet(packet)
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Attachment changed'):
        module.check_packet(packet)


def test_partial_export_is_not_truth_or_a_completed_vote(module, tmp_path):
    image = tmp_path / 'source.jpg'
    image.write_bytes(b'untouched')
    module.dump(tmp_path / 'private_pairs.json', [dict(case_id='S001', stable_id='sd_original', original=str(image))])
    module.dump(tmp_path / 'manifest.json', dict(source_sha256={str(image): module.sha(image)}))
    batch = tmp_path / 'claude/batch_01'
    batch.mkdir(parents=True)
    module.dump(batch / 'validated.json', dict(model='real-model', items=[item()]))
    summary = module.export(tmp_path)
    assert summary['comparison']['incomplete'] == 1
    assert summary['comparison']['unanimous'] == 0
    assert summary['all_selected_sources_unchanged']
    assert image.read_bytes() == b'untouched'
    assert 'AI_reference_not_human_truth' in (tmp_path / 'labels_claude.csv').read_text(encoding='utf-8-sig')
    assert 'pending_or_failed' in (tmp_path / 'labels_gpt.csv').read_text(encoding='utf-8-sig')
    assert 'accuracy' not in summary


def test_transport_failure_stops_provider_without_fabricating_labels(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'check_packet', lambda packet: None)
    def blocked(*args):
        raise ValueError('quota exhausted')
    monkeypatch.setattr(module, 'claude', blocked)
    packet = dict(name='batch_01', images={}, prompt_sha256='test')
    status = module.run_one('claude', packet, tmp_path)
    assert not status['valid']
    assert not (tmp_path / 'claude/batch_01/validated.json').exists()
    assert json.loads((tmp_path / 'claude/batch_01/status.json').read_text())['error'].endswith('quota exhausted')


def test_explicit_retry_preserves_failure_and_never_repeats_success(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'check_packet', lambda packet: None)
    old = tmp_path / 'gpt/batch_01'
    old.mkdir(parents=True)
    module.dump(old / 'status.json', dict(valid=False, error='quota exhausted'))
    calls = []
    def response(*args):
        calls.append(True)
        return {'items': [item()]}, 'gpt-6.1-sol', {}
    monkeypatch.setattr(module, 'gpt', response)
    packet = dict(name='batch_01', ids=['S001'], images={}, prompt_sha256='test')
    assert not module.run_one('gpt', packet, tmp_path)['valid']
    assert calls == []
    assert module.run_one('gpt', packet, tmp_path, retry_failed=True)['valid']
    assert len(list((tmp_path / 'previous_attempts/gpt').glob('*/status.json'))) == 1
    assert module.run_one('gpt', packet, tmp_path, retry_failed=True)['valid']
    assert len(calls) == 1
