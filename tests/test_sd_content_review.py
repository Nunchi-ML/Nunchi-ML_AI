import csv
import importlib
import json
from pathlib import Path

from PIL import Image
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'notebooks/experiments/sun_visible_probe'))
    return importlib.import_module('sd_content_review')


def fixture_data(module, tmp_path, group='20261103'):
    root = tmp_path / 'raw'
    (root / 'images').mkdir(parents=True)
    rows = []
    for i in range(2):
        photo = root / 'images' / f'{i}.jpg'
        Image.new('RGB', (32, 18), (80+i, 90, 100)).save(photo)
        rows.append(dict(stable_id=f'sd_{i:016x}', image=f'images/{i}.jpg', group_id=group,
                         clip_id=f'clip{i}', second='0', sha256=module.sha(photo), glare_label_final=''))
    source, previous = tmp_path / 'development.csv', tmp_path / 'previous.csv'
    module.write_csv(source, rows, list(rows[0]))
    module.write_csv(previous, rows[:1], list(rows[0]))
    return root, source, previous, rows


def test_reserved_day_and_path_escape_rejected(module, tmp_path):
    root, source, previous, _ = fixture_data(module, tmp_path, '20261105')
    with pytest.raises(ValueError, match='Reserved date'):
        module.prepare(root, source, previous, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
    with pytest.raises(ValueError, match='inside root/images'):
        module.safe_source(root, '../outside.jpg')


def test_screen_pack_excludes_old_ids_and_preserves_sources(module, tmp_path):
    root, source, previous, rows = fixture_data(module, tmp_path)
    before = {p:module.sha(p) for p in [source, previous, *root.glob('images/*.jpg')]}
    out = tmp_path / 'output'
    result = module.prepare(root, source, previous, out)
    assert result == dict(new_images=1, pages=1)
    mapping = module.read_csv(out / 'private_mapping.csv')
    assert mapping[0]['stable_id'] == rows[1]['stable_id']
    assert all(module.sha(p) == digest for p, digest in before.items())
    with pytest.raises(ValueError, match='new private output'):
        module.prepare(root, source, previous, out)


def test_finalize_rejects_missing_page_duplicate_and_unviewed_observations(module, tmp_path):
    root, source, previous, _ = fixture_data(module, tmp_path)
    out, observed = tmp_path / 'output', tmp_path / 'observed.json'
    module.prepare(root, source, previous, out)
    module.dump(observed, [])
    with pytest.raises(ValueError, match='every viewed page'):
        module.finalize(out, observed)
    base = dict(page=1, viewed=False, note='not viewed', flagged=[])
    module.dump(observed, [base])
    with pytest.raises(ValueError, match='Missing page review'):
        module.finalize(out, observed)
    base.update(viewed=True, flagged=[dict(case_id='D0001', reason='solar candidate')]*2)
    module.dump(observed, [base])
    with pytest.raises(ValueError, match='Duplicate content candidate'):
        module.finalize(out, observed)
    assert not (out / 'content_candidates').exists()


def test_final_candidates_are_original_bytes_and_not_severity_labels(module, tmp_path):
    root, source, previous, rows = fixture_data(module, tmp_path)
    out, observed = tmp_path / 'output', tmp_path / 'observed.json'
    module.prepare(root, source, previous, out)
    module.dump(observed, [dict(page=1, viewed=True, note='Viewed one remaining image',
                               flagged=[dict(case_id='D0001', reason='<script>uncertain</script>')])])
    report = module.finalize(out, observed)
    assert report['severity_labels_created'] == 0
    assert not report['reserved_day_used']
    assert report['content_candidates'] == 1
    copied = out / 'content_candidates/images' / (rows[1]['stable_id'] + '.jpg')
    assert module.sha(copied) == rows[1]['sha256']
    assert '&lt;script&gt;' in (out / 'content_candidates/review.html').read_text(encoding='utf-8')
    assert all(not r['glare_label_final'] for r in module.read_csv(source))
