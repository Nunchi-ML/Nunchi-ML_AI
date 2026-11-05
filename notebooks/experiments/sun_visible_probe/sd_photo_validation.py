"""Fixed training624 -> SD GPT-reference development comparison, without G.

Keep initial300 and content-enriched69 separate. Predict every photo before
opening SD reference grades. No SD fitting/selection, relabeling or deployment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd

import road_visibility_probe as R
import forward_detail_features as F
import photo_regularization_probe as P
from rel_features import rel_array
from nunchi_ml.glare.features import extract_array, FEATURES
from nunchi_ml.glare.evaluation import evaluate_grades


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def macro_rows(y, predictions):
    y, predictions = np.asarray(y, int), np.asarray(predictions, int)
    scores = []
    for c in range(3):
        tp = ((predictions == c) & (y == c)).sum(axis=1)
        denominator = (predictions == c).sum(axis=1) + (y == c).sum()
        scores.append(np.divide(2*tp, denominator, out=np.zeros(len(predictions)), where=denominator != 0))
    return np.mean(scores, axis=0)


def fit_rule(train, columns):
    """Same19 quantiles/two directions as old single-feature rules; training only."""
    best = None
    for col in columns:
        v = train[col].to_numpy(float)
        if not np.isfinite(v).all():
            raise ValueError('Finite training rule features required')
        qs = np.unique(np.quantile(v, np.linspace(.05, .95, 19)))
        choices = [(float(a), float(b), sign) for i, a in enumerate(qs) for b in qs[i+1:] for sign in (1, -1)]
        if not choices:
            continue
        predictions = np.asarray([np.digitize(v, [a, b]) if sign == 1 else np.digitize(-v, [-b, -a]) for a, b, sign in choices])
        scores = macro_rows(train.label, predictions)
        k = int(np.argmax(scores))
        if best is None or scores[k] > best['training_macro_f1']:
            a, b, sign = choices[k]
            best = dict(feature=col, low=a, high=b, sign=sign, training_macro_f1=float(scores[k]))
    if best is None:
        raise ValueError('No nonconstant rule feature')
    return best


def apply_rule(test, rule):
    v = test[rule['feature']].to_numpy(float)
    if not np.isfinite(v).all():
        raise ValueError('Finite evaluation rule features required')
    a, b = rule['low'], rule['high']
    return np.digitize(v, [a, b]) if rule['sign'] == 1 else np.digitize(-v, [-b, -a])


def join_reference(metadata, labels):
    """Full coverage first; abstentions remain explicit and never become0."""
    if (metadata.stable_id.duplicated().any() or labels.stable_id.duplicated().any()
            or metadata.stable_id.isna().any() or labels.stable_id.isna().any()
            or set(metadata.stable_id) != set(labels.stable_id)):
        raise ValueError('Exact unique reference coverage required')
    if not labels.status.eq('complete').all() or not labels.grade.isin(['0', '1', '2', 'abstain']).all():
        raise ValueError('Complete GPT references with explicit abstentions required')
    joined = metadata.merge(labels[['stable_id', 'grade']], on='stable_id', validate='one_to_one', sort=False)
    joined['accepted_reference'] = joined.grade.ne('abstain')
    joined['label'] = pd.to_numeric(joined.grade.where(joined.accepted_reference), errors='coerce')
    return joined


def verify_reproduction(actual, frozen):
    keys = ['id', 'method']
    if actual.duplicated(keys).any() or frozen.duplicated(keys).any():
        raise ValueError('Unique reproduction keys required')
    a, b = [x.set_index(keys).sort_index() for x in (actual, frozen)]
    if not a.index.equals(b.index) or not np.array_equal(a.pred, b.pred):
        raise ValueError('Frozen human50 predictions changed')
    if all(c in b for c in ['p0', 'p1', 'p2']) and not np.allclose(a[['p0', 'p1', 'p2']], b[['p0', 'p1', 'p2']], atol=1e-10, rtol=0):
        raise ValueError('Frozen human50 probabilities changed')


def paired_intervals(frame, names, baseline='rule_all_features', repeats=2000):
    """Fixed-prediction filename-date resamples, separate for each SD subset."""
    records = []
    for subset, part in frame.groupby('subset', sort=False):
        groups = sorted(part.group_id.unique())
        counts = np.zeros((len(groups), len(names), 3, 3), int)
        for g, group in enumerate(groups):
            local = part[part.group_id.eq(group)]
            for n, name in enumerate(names):
                np.add.at(counts[g, n], (local.label.to_numpy(int), local[name].to_numpy(int)), 1)
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(repeats):
            cm = counts[rng.integers(0, len(groups), len(groups))].sum(axis=0)
            diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
            den = cm.sum(axis=-2)+cm.sum(axis=-1)
            f = np.divide(2*diagonal, den, out=np.zeros(diagonal.shape, float), where=den != 0).mean(axis=-1)
            draws.append(f-f[names.index(baseline)])
        for n, name in enumerate(names):
            records.append(dict(subset=subset, method=name, baseline=baseline, filename_date_groups=len(groups),
                                low=float(np.quantile(np.asarray(draws)[:, n], .025)), high=float(np.quantile(np.asarray(draws)[:, n], .975)),
                                scope='Fixed predictions/filename-date composition only; unverified clocks/trips, no refit'))
    return records


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(data):
        raise ValueError('Use a new private output directory')
    old, detail = data/'road_visibility_20261104/run30', data/'forward_detail_20261105'
    sources = [Path(__file__), Path(R.__file__), Path(R.C.__file__), Path(R.R.__file__), Path(F.__file__),
               Path(P.__file__), Path(extract_array.__code__.co_filename), Path(rel_array.__code__.co_filename),
               Path(evaluate_grades.__code__.co_filename), Path(R.lightgbm_model.__code__.co_filename)]
    inputs = [old/'training_features.csv', old/'human_features.csv', old/'predictions.csv',
              detail/'train_features.csv', detail/'human50_features.csv', detail/'photo_predictions.csv',
              data/'photo_regularization_20261105/choices_before_evaluation.json',
              data/'photo_regularization_20261105/predictions.csv', data/'sd_curation_20261105/development_candidates.csv']
    packs = [(data/'sd_ai_labels_20261105', 'initial300'), (data/'sd_supplement_labels_20261105', 'content69')]
    inputs += [folder/name for folder, _ in packs for name in ['manifest.json', 'private_pairs.json', 'labels_gpt.csv']]
    hashes = {str(p): sha(p) for p in inputs+sources}
    out.mkdir()
    cols = dict(screen_lgb=FEATURES, rel_logistic=R.P.REL_R,
                screen_road_lgb=FEATURES+R.R.FEATURES, rel_road_logistic=R.P.REL_R+R.R.FEATURES,
                screen_road_detail_lgb=FEATURES+R.R.FEATURES+F.FEATURES,
                rel_road_detail_logistic=R.P.REL_R+R.R.FEATURES+F.FEATURES,
                selected_detail_lgb=FEATURES+R.R.FEATURES+F.FEATURES)
    all_features = list(dict.fromkeys(FEATURES+R.P.REL_R+R.R.FEATURES+F.FEATURES))
    protocol = dict(status='frozen_before_SD_extraction_prediction_scoring', training='Original624/27capture/labels unchanged; noSD fitting',
        reference='Existing GPT numerical grades only; abstentions excluded explicitly, never human truth',
        evaluation='initial300 and content69 separate; known development selection/62shared clips, no final independent test',
        models=cols, decisions='LGB argmax; balanced LR alert>=.5 then p2>=p1',
        rules='R1mean_all,R2white_ratio,R3bestRelative12,R4bestAll54;19training quantiles/two directions/first tie',
        controls=['always_normal','always_alert'], probability_policy='Raw ML probabilities uncalibrated; hard rules no fabricated probabilities/AUC',
        intervals='2000 seed0 filename-date resamples within subset; fixed predictions only',
        no_geometry_sensor_or_neural=True, no_SD_label_selection=True, hashes=hashes,
        versions={n: importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','lightgbm','opencv-python-headless']})
    dump(out/'specification_before_run.json', protocol)
    for i, path in enumerate(sources):
        shutil.copyfile(path, out/f'source_{i:02d}_{path.name}')
    train = pd.read_csv(inputs[0], dtype={'id':str}).merge(pd.read_csv(inputs[3], dtype={'id':str}), on='id', validate='one_to_one')
    human = pd.read_csv(inputs[1], dtype={'id':str}).merge(pd.read_csv(inputs[4], dtype={'id':str}), on='id', validate='one_to_one')
    if len(train) != 624 or train.group.nunique() != 27 or len(human) != 50 or train.image_sha256.duplicated().any():
        raise ValueError('Frozen624/27capture and human50 reproduction required')
    for path, expected in zip(R.C.image_paths(train, data), train.image_sha256):
        if sha(path) != expected:
            raise ValueError('Original training photo changed')
        hashes[str(path)] = expected
    development = pd.read_csv(inputs[8], dtype={'group_id':str})
    records = []
    for folder, subset in packs:
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if any(sha(path) != expected for path, expected in manifest['source_sha256'].items()):
            raise ValueError('Frozen SD labeling sources changed')
        for row in json.loads((folder/'private_pairs.json').read_text(encoding='utf-8')):
            records.append(dict(id=row['stable_id'], stable_id=row['stable_id'], image_sha256=row['sha256'], image=row['original'], subset=subset))
    metadata = pd.DataFrame(records).merge(development[['stable_id','group_id','clip_id','filename_datetime']], on='stable_id', how='left', validate='one_to_one')
    if (len(metadata) != 369 or metadata.group_id.isna().any() or metadata.group_id.eq('20261105').any()
            or metadata.image_sha256.duplicated().any() or set(metadata.image_sha256) & set(train.image_sha256)):
        raise ValueError('Distinct nonreserved SD development photos without training overlap required')
    def extract(row):
        raw = Path(row.image).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row.image_sha256:
            raise ValueError('SD image bytes changed')
        image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('SD image decode failed')
        return dict(id=row.id, **extract_array(image), **{'r_'+k:v for k,v in rel_array(image).items()}, **R.R.extract_array(image), **F.extract_array(image))
    with ThreadPoolExecutor(max_workers=4) as pool:
        fresh = pd.DataFrame(pool.map(extract, metadata.itertuples()))
    test = metadata.merge(fresh, on='id', validate='one_to_one')
    if not all(np.isfinite(x[all_features].to_numpy(float)).all() for x in [train,human,test]):
        raise ValueError('Finite features required, no row dropping')
    test.to_csv(out/'SD_features.csv',index=False,encoding='utf-8-sig')
    hashes.update({str(Path(r.image)):r.image_sha256 for r in metadata.itertuples()})
    rules = {name:fit_rule(train, columns) for name,columns in [('rule_brightness',['mean_all']),('rule_white',['white_ratio']),('rule_relative',R.P.REL_R),('rule_all_features',all_features)]}
    selected = json.loads(inputs[6].read_text(encoding='utf-8'))['detail']
    dump(out/'choices_before_SD_prediction.json', dict(rules=rules, selected_detail_parameters=selected,
        source='Original624 training only / previously frozen settings, before opening SD label rows'))
    combined = pd.concat([human.assign(subset='human50_reproduction'),test],ignore_index=True)
    prediction, probability = {}, {}
    for name, columns in cols.items():
        if name.endswith('lgb'):
            model = P.model(selected) if name == 'selected_detail_lgb' else R.lightgbm_model()
            model.fit(train[columns],train.label)
            p = np.zeros((len(combined),3));p[:,model.classes_.astype(int)] = model.predict_proba(combined[columns])
            pred = p.argmax(1)
        else:
            p = R.C.probabilities(train,combined,columns);pred = R.C.decision(p)
        R.validate_probabilities(p)
        probability[name],prediction[name] = p,pred
        print(name,'fit624/predict419 complete',flush=True)
    for name,rule in rules.items():
        prediction[name] = apply_rule(combined,rule)
    prediction.update(always_normal=np.zeros(len(combined),int),always_alert=np.ones(len(combined),int))
    # Reproduce all seven pre-existing human50 methods before opening SD grades.
    frozen = pd.read_csv(old/'predictions.csv',dtype={'id':str})
    frozen = frozen[frozen.test.eq('human50_photo_only') & frozen.method.isin(list(cols)[:4])]
    actual = pd.DataFrame([dict(id=cid,method=name,pred=int(prediction[name][i]),**{f'p{k}':float(probability[name][i,k]) for k in range(3)})
                           for name in list(cols)[:4] for i,cid in enumerate(human.id)])
    verify_reproduction(actual,frozen)
    fd = pd.read_csv(detail/'photo_predictions.csv',dtype={'id':str}).set_index('id').loc[human.id]
    selected_frozen = pd.read_csv(data/'photo_regularization_20261105/predictions.csv',dtype={'id':str}).set_index('id').loc[human.id]
    for name in ['screen_road_detail_lgb','rel_road_detail_logistic']:
        if not np.array_equal(prediction[name][:50],fd[name]):
            raise ValueError('Detail human50 reproduction changed')
    if not np.array_equal(prediction['selected_detail_lgb'][:50],selected_frozen.selected_detail):
        raise ValueError('Selected detail human50 reproduction changed')
    # Save predictions before reading SD grade rows.
    result = metadata.copy()
    for name,pred in prediction.items():
        result[name] = pred[50:]
    result.to_csv(out/'predictions_before_reference.csv',index=False,encoding='utf-8-sig')
    references = pd.concat([pd.read_csv(folder/'labels_gpt.csv',dtype=str) for folder,_ in packs],ignore_index=True)
    result = join_reference(result,references)
    result.to_csv(out/'reference_coverage.csv',index=False,encoding='utf-8-sig')
    scores, detailed = [], []
    for subset,part in result[result.accepted_reference].groupby('subset',sort=False):
        indices = part.index.to_numpy()+50
        for name in prediction:
            p = probability[name][indices] if name in probability else None
            scores.append(dict(subset=subset,method=name,reference_kind='GPT_reference_not_human_truth',**evaluate_grades(part.label.to_numpy(int),part[name].to_numpy(int),p)))
    for i,row in result.iterrows():
        for name in prediction:
            p = probability[name][i+50] if name in probability else [None]*3
            detailed.append(dict(stable_id=row.stable_id,subset=row.subset,group_id=row.group_id,clip_id=row.clip_id,
                reference_grade=row.grade,accepted_reference=bool(row.accepted_reference),method=name,pred=int(row[name]),
                p0=p[0],p1=p[1],p2=p[2],actual_A=None,actual_B=None,
                decision_source='Frozen624 fit/previous settings; no SD selection',reference_kind='GPT_reference_not_human_truth'))
    pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(detailed).to_csv(out/'predictions_long.csv',index=False,encoding='utf-8-sig')
    intervals = paired_intervals(result[result.accepted_reference],list(prediction))
    pd.DataFrame(intervals).to_csv(out/'paired_date_intervals.csv',index=False,encoding='utf-8-sig')
    dump(out/'input_hashes_after_run.json',hashes)
    if any(sha(path) != value for path,value in hashes.items()):
        raise ValueError('Input/source changed during comparison')
    report = dict(status='complete',training_rows=624,training_groups=27,reproduced_human_predictions=350,
        SD_images=369,coverage=result.groupby('subset').accepted_reference.agg(['size','sum']).to_dict('index'),
        rules=rules,scores=scores,paired_intervals=intervals,all_inputs_unchanged=True,
        limits=['GPT references, not human accuracy; no independent final test','initial300/content69 kept separate; shared62clips',
                'Only1/3 severe references; date clusters and clocks unverified','No SD fitting/threshold choice, G/sensor input, neural model or deployment change',
                'Raw ML probabilities uncalibrated; hard rules no fabricated probability metrics'])
    dump(out/'report.json',report)
    columns=['subset','method','macro_f1','f1_1','recall_2','count_2_to_0','accuracy','alert_roc_auc','alert_average_precision']
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>SD GPT-reference photo comparison</title><style>body{font:16px system-ui;margin:28px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}</style><h1>SD GPT-reference development comparison</h1><p>Original624 training only; GPT reference, not human ground truth. Initial300 and enriched69 separate. No G/sensor or final model selection.</p>'+pd.DataFrame(scores).reindex(columns=columns).round(4).to_html(index=False),encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    report = run(args.data,args.out)
    print(json.dumps({k:report[k] for k in ['status','coverage','reproduced_human_predictions']},ensure_ascii=True))
