"""User-requested addition of frozen historical photo-weight candidates.

An explicitly subsequent comparison, after the seven-model SD study. No SD
selection: training-selected .481699 and human50-observed max .483472 configs
are both retained. Do not overwrite the initial experiment or predictions.
"""
import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import sd_photo_validation as S
import photo_weight_probe as W


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(data):
        raise ValueError('Use new private output')
    base, old, detail, weighted = [data/n for n in ['sd_photo_validation_20261106','road_visibility_20261104/run30','forward_detail_20261105','photo_weights_20261105']]
    files = [base/'SD_features.csv',base/'reference_coverage.csv',base/'metrics.csv',old/'training_features.csv',
             old/'human_features.csv',detail/'train_features.csv',detail/'human50_features.csv',
             weighted/'predictions.csv',weighted/'scores.csv',weighted/'choices_before_evaluation.json',
             Path(__file__),Path(S.__file__),Path(W.__file__),Path(W.P.__file__)]
    hashes = {str(p):S.sha(p) for p in files}
    out.mkdir()
    candidates = dict(training_selected_weight='capture_sqrt',historical_human50_max='capture_inverse')
    S.dump(out/'protocol_before_fit.json',dict(status='subsequent_user_requested_comparison',candidates=candidates,
        parameters=W.PARAMETERS['regularized'],columns=S.FEATURES+S.R.R.FEATURES+S.F.FEATURES,
        selection_source='Frozen original624 training selection and earlier human50 max; neither selected on SD. First SD outcomes already known.',
        limits='GPT reference; shared development clips/rare severe, not final human accuracy or G-model comparison',hashes=hashes))
    shutil.copyfile(__file__,out/'source_before_fit.py')
    train = pd.read_csv(files[3],dtype={'id':str}).merge(pd.read_csv(files[5],dtype={'id':str}),on='id',validate='one_to_one')
    human = pd.read_csv(files[4],dtype={'id':str}).merge(pd.read_csv(files[6],dtype={'id':str}),on='id',validate='one_to_one')
    test = pd.read_csv(files[0],dtype={'group_id':str})
    ref = pd.read_csv(files[1],dtype={'group_id':str})
    frozen = pd.read_csv(files[7],dtype={'id':str}).set_index('id').loc[human.id]
    if len(train)!=624 or train.group.nunique()!=27 or len(human)!=50 or len(test)!=369 or not test.id.equals(ref.id):
        raise ValueError('Frozen training/reproduction/SD rows required')
    columns = S.FEATURES+S.R.R.FEATURES+S.F.FEATURES
    combined = pd.concat([human,test],ignore_index=True)
    scores,records = [],[]
    for name,policy in candidates.items():
        model = W.P.model(W.PARAMETERS['regularized']).set_params(class_weight=None)
        model.fit(train[columns],train.label,sample_weight=W.training_weights(train,policy))
        p = np.zeros((len(combined),3));p[:,model.classes_.astype(int)] = model.predict_proba(combined[columns])
        S.R.validate_probabilities(p);pred = p.argmax(1)
        previous = 'detail_regularized_'+policy
        if not np.array_equal(pred[:50],frozen[previous]):
            raise ValueError('Historical weighted human50 predictions changed')
        ref[name] = pred[50:]
        for subset,part in ref[ref.accepted_reference].groupby('subset',sort=False):
            idx = part.index.to_numpy()+50
            scores.append(dict(subset=subset,method=name,policy=policy,**S.evaluate_grades(part.label.to_numpy(int),part[name].to_numpy(int),p[idx])))
        for i,row in ref.iterrows():
            records.append(dict(stable_id=row.stable_id,subset=row.subset,group_id=row.group_id,clip_id=row.clip_id,
                reference_grade=row.grade,accepted_reference=bool(row.accepted_reference),method=name,pred=int(pred[i+50]),
                p0=p[i+50,0],p1=p[i+50,1],p2=p[i+50,2],actual_A=None,actual_B=None,
                training_source='original624',selection_source='frozen historical settings; added after initial SD results'))
    pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(records).to_csv(out/'predictions_long.csv',index=False,encoding='utf-8-sig')
    ref.to_csv(out/'comparison_with_original_methods.csv',index=False,encoding='utf-8-sig')
    pairs=[]
    for baseline in ['screen_lgb','rule_brightness']:
        pairs += S.paired_intervals(ref[ref.accepted_reference],list(candidates)+[baseline],baseline=baseline)
    pd.DataFrame(pairs).to_csv(out/'paired_date_intervals.csv',index=False,encoding='utf-8-sig')
    if any(S.sha(p)!=value for p,value in hashes.items()):
        raise ValueError('Frozen sources/inputs changed')
    report = dict(status='complete',reproduced_human50_predictions=100,scores=scores,all_inputs_unchanged=True,
                  candidates=candidates,not_G_model=True,not_SD_training=True,subsequent_after_initial_scores=True)
    S.dump(out/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.data,args.out)
