"""SD fixed-error audit and one-factor photo+observed-weather transfer study.

All learning uses original624 only. Cloud/sunshine are regional historical
observations, not full G geometry, on-vehicle illuminance or live availability.
"""
import argparse
import html
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

import sd_photo_validation as S

WEATHER = ['cloud','sunshine']


def attach_weather(frame, weather):
    if frame.stable_id.duplicated().any() or weather.stable_id.duplicated().any():
        raise ValueError('Unique image/weather keys required')
    cols = ['stable_id',*WEATHER,'weather_observed_at','weather_requested_hour','weather_age_minutes','weather_availability_verified']
    result = frame.merge(weather[cols],on='stable_id',how='left',sort=False,validate='one_to_one')
    if (len(result)!=len(frame) or not result.stable_id.equals(frame.stable_id.reset_index(drop=True))
            or result.weather_observed_at.isna().any() or result.weather_requested_hour.isna().any()
            or result.weather_age_minutes.isna().any() or result.weather_age_minutes.lt(0).any()):
        raise ValueError('Exact nonfuture weather coverage required; no fallback')
    observed=pd.to_datetime(result.weather_observed_at)
    requested=pd.to_datetime(result.weather_requested_hour)
    captured=pd.to_datetime(result.filename_datetime)
    if not observed.equals(requested) or not requested.equals(captured.dt.floor('h')):
        raise ValueError('Observation must match preceding capture hour exactly')
    if (not np.isfinite(result[WEATHER].to_numpy(float)).all()
            or not result.cloud.between(0,10).all() or not result.sunshine.between(0,1).all()):
        raise ValueError('Complete bounded cloud/sunshine required; no zero imputation')
    return result


def error_flags(truth, pred):
    y,p = np.asarray(truth,int),np.asarray(pred,int)
    if y.shape!=p.shape or not np.isin(y,[0,1,2]).all() or not np.isin(p,[0,1,2]).all():
        raise ValueError('Matched valid grades required')
    return np.where((y==0)&(p>=1),'FP_GPT',np.where((y>=1)&(p==0),'FN_GPT',np.where(y!=p,'grade_disagreement','agreement')))


def select_cases(frame):
    """Fixed 2FP/2FN/normal+positive controls per subset, distinct clips globally."""
    selected,used = [],set()
    for subset in ['initial300','content69']:
        part=frame[frame.subset.eq(subset)]
        for kind,n in [('FP_GPT',2),('FN_GPT',2),('normal_control',1),('positive_control',1)]:
            if kind=='normal_control':local=part[part.label.eq(0)&part.screen_lgb.eq(0)]
            elif kind=='positive_control':local=part[part.label.ge(1)&part.screen_lgb.eq(part.label)]
            else:local=part[part.error.eq(kind)]
            local=local.assign(priority=local.p0 if kind=='FN_GPT' else 1-local.p0)
            local=local.sort_values(['priority','stable_id'],ascending=[False,True])
            picked=0
            for row in local.to_dict('records'):
                if row['clip_id'] in used:continue
                selected.append({**row,'review_kind':kind});used.add(row['clip_id']);picked+=1
                if picked==n:break
            if picked!=n:raise ValueError('Insufficient distinct clips for fixed review design')
    return pd.DataFrame(selected)


def run(data,out):
    data,out=Path(data).resolve(),Path(out).resolve()
    if out.exists() or not out.is_relative_to(data):raise ValueError('Use new private output')
    base,old,detail,weather=[data/n for n in ['sd_photo_validation_20261106','road_visibility_20261104/run30','forward_detail_20261105','sd_weather_20261106']]
    paths=[base/'SD_features.csv',base/'reference_coverage.csv',base/'predictions_long.csv',base/'metrics.csv',
           old/'training_features.csv',detail/'train_features.csv',weather/'frame_weather_alignment.csv',weather/'report.json',
           data/'sd_ai_labels_20261105/labels_gpt.csv',data/'sd_supplement_labels_20261105/labels_gpt.csv',
           Path(__file__),Path(S.__file__),Path(S.R.lightgbm_model.__code__.co_filename)]
    hashes={str(p):S.sha(p) for p in paths}
    out.mkdir()
    S.dump(out/'specification_before_run.json',dict(status='frozen_before_weather_candidate_fit',hashes=hashes,
        hypothesis='Add only observed regional cloud/sunshine to frozen screenLGB; fixed original624 training, same model settings',
        methods=['screen_reproduced','screen_weather_lgb','weather_only_lgb','rule_weather_only','rule_all56'],
        reference='Existing GPT grades; initial300/content69 separate, abstentions excluded, no relabeling/SD training',
        audit='Existing screen errors by GPT cause/evidence/hour; top-confidence2FP/2FN+2controls per subset,12globallydistinct clips',
        distribution='Train versus each SD subset feature quartiles/p01-p99/outside rates; descriptive only, not refitting',
        bootstrap='2000 seed0 filename-date resamples within each subset, baseline screen_reproduced',
        limits='Regional historical ASOS proxy, not fullG/cameraGPS/heading, on-car lux or verified live availability. Design after initial SD results.'))
    shutil.copyfile(__file__,out/'source_before_run.py')
    train=pd.read_csv(paths[4],dtype={'id':str}).merge(pd.read_csv(paths[5],dtype={'id':str}),on='id',validate='one_to_one')
    features=pd.read_csv(paths[0],dtype={'group_id':str})
    ref=pd.read_csv(paths[1],dtype={'group_id':str})
    test=attach_weather(features,pd.read_csv(paths[6],dtype={'group_id':str}))
    if (len(train)!=624 or train.group.nunique()!=27 or len(test)!=369 or not test.id.equals(ref.id)
            or set(train.image_sha256)&set(test.image_sha256)):
        raise ValueError('Frozen624/27capture and369SD without overlap required')
    columns=list(dict.fromkeys(S.FEATURES+S.R.P.REL_R+S.R.R.FEATURES+S.F.FEATURES+WEATHER))
    if not np.isfinite(train[columns].to_numpy(float)).all():raise ValueError('Finite original training inputs required')
    for r in test.itertuples():
        if S.sha(r.image)!=r.image_sha256:raise ValueError('Original SD image changed')
    labels=pd.concat([pd.read_csv(p,dtype=str) for p in paths[8:10]],ignore_index=True)
    if set(labels.stable_id)!=set(test.stable_id):raise ValueError('GPT reference coverage changed')
    comparison=ref.merge(labels[['stable_id','cause','sun_evidence','visibility']],on='stable_id',validate='one_to_one',sort=False)
    baseline=pd.read_csv(paths[2],dtype={'group_id':str});baseline=baseline[baseline.method.eq('screen_lgb')].set_index('stable_id').loc[test.stable_id]
    comparison['p0']=baseline.p0.to_numpy();accepted=comparison[comparison.accepted_reference].copy()
    accepted['error']=error_flags(accepted.label.to_numpy(int),accepted.screen_lgb.to_numpy(int))
    accepted['hour']=pd.to_datetime(accepted.filename_datetime).dt.hour
    accepted.groupby(['subset','error','cause','sun_evidence','hour'],dropna=False).size().rename('rows').reset_index().to_csv(out/'error_context.csv',index=False,encoding='utf-8-sig')
    selected=select_cases(accepted)
    selected.to_csv(out/'review_12.csv',index=False,encoding='utf-8-sig')
    sheet=Image.new('RGB',(1280,792),'#eeeeee');draw=ImageDraw.Draw(sheet)
    cards=[]
    for i,r in enumerate(selected.itertuples()):
        x,y=(i%4)*320,(i//4)*264
        with Image.open(r.image) as image:sheet.paste(ImageOps.contain(image.convert('RGB'),(320,220)),(x,y+40))
        draw.text((x+4,y+2),f'{i+1:02d} {r.subset} {r.review_kind}',fill='black')
        draw.text((x+4,y+20),f'GPT {int(r.label)} / ML {int(r.screen_lgb)}',fill='black')
        cards.append(f'<article><h2>{i+1:02d} {html.escape(r.review_kind)} · GPT {int(r.label)} / ML {int(r.screen_lgb)}</h2><img width="960" src="{Path(r.image).as_uri()}"><p>GPT cause: {html.escape(r.cause)} / evidence: {html.escape(r.sun_evidence)}</p></article>')
    sheet.save(out/'review_12_sheet.jpg',quality=92)
    (out/'review_12.html').write_text('<!doctype html><meta charset="utf-8"><title>SD error audit</title><h1>GPT-reference disagreements and controls</h1><p>Posthoc GPT-family auxiliary audit, not independent human truth. Original grades/predictions preserved.</p>'+''.join(cards),encoding='utf-8')
    distribution=[]
    for col in columns:
        tr=train[col].to_numpy(float);lo,hi=np.quantile(tr,[.01,.99]);q1,med,q3=np.quantile(tr,[.25,.5,.75])
        for subset,part in test.groupby('subset',sort=False):
            v=part[col].to_numpy(float)
            distribution.append(dict(subset=subset,feature=col,train_p01=lo,train_median=med,train_p99=hi,SD_median=float(np.median(v)),
                median_shift_train_IQR=float((np.median(v)-med)/(q3-q1)) if q3>q1 else None,
                outside_training_p01_p99=float(((v<lo)|(v>hi)).mean())))
    pd.DataFrame(distribution).to_csv(out/'feature_distribution.csv',index=False,encoding='utf-8-sig')
    prediction,probability={},{}
    for name,cols in [('screen_reproduced',S.FEATURES),('screen_weather_lgb',S.FEATURES+WEATHER),('weather_only_lgb',WEATHER)]:
        model=S.R.lightgbm_model().fit(train[cols],train.label)
        p=np.zeros((len(test),3));p[:,model.classes_.astype(int)]=model.predict_proba(test[cols]);S.R.validate_probabilities(p)
        probability[name],prediction[name]=p,p.argmax(1)
    if (not np.array_equal(prediction['screen_reproduced'],baseline.pred.to_numpy())
            or not np.allclose(probability['screen_reproduced'],baseline[['p0','p1','p2']].to_numpy(float),atol=1e-10,rtol=0)):
        raise ValueError('Original369 screen predictions/probabilities changed')
    rules={name:S.fit_rule(train,cols) for name,cols in [('rule_weather_only',WEATHER),('rule_all56',columns)]}
    S.dump(out/'training_rules.json',rules)
    for name,rule in rules.items():prediction[name]=S.apply_rule(test,rule)
    result=ref.copy()
    for name,pred in prediction.items():result[name]=pred
    scores,records,tradeoffs=[],[],[]
    for subset,part in result[result.accepted_reference].groupby('subset',sort=False):
        idx=part.index.to_numpy();truth=part.label.to_numpy(int)
        for name,pred in prediction.items():
            p=probability[name][idx] if name in probability else None
            scores.append(dict(subset=subset,method=name,reference_kind='GPT_reference_not_human_truth',**S.evaluate_grades(truth,pred[idx],p)))
            positive=truth>=1;alert=pred[idx]>=1
            tradeoffs.append(dict(subset=subset,method=name,alert_misses=int((~alert[positive]).sum()),positive_support=int(positive.sum()),
                false_alerts=int(alert[~positive].sum()),normal_support=int((~positive).sum()),
                severe_hits=int(((truth==2)&(pred[idx]==2)).sum()),severe_support=int((truth==2).sum())))
    for i,row in result.iterrows():
        for name,pred in prediction.items():
            p=probability[name][i] if name in probability else [None]*3
            records.append(dict(stable_id=row.stable_id,subset=row.subset,group_id=row.group_id,clip_id=row.clip_id,
                reference_grade=row.grade,accepted_reference=bool(row.accepted_reference),method=name,pred=int(pred[i]),p0=p[0],p1=p[1],p2=p[2],
                actual_A=None,actual_B=None,training_source='Original624 only',selection_source='Frozen one-factor weather comparison; no SD tuning',
                weather_source='RegionalASOS143/historical previous-hour, publication unverified'))
    result.to_csv(out/'comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(records).to_csv(out/'predictions_long.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(tradeoffs).to_csv(out/'alert_tradeoffs.csv',index=False,encoding='utf-8-sig')
    intervals=S.paired_intervals(result[result.accepted_reference],list(prediction),baseline='screen_reproduced')
    pd.DataFrame(intervals).to_csv(out/'paired_date_intervals.csv',index=False,encoding='utf-8-sig')
    if any(S.sha(p)!=v for p,v in hashes.items()):raise ValueError('Frozen source/input changed')
    report=dict(status='complete',training_rows=624,SD_images=369,reproduced_predictions_and_probabilities=369,
        review_cases=12,review_distinct_clips=int(selected.clip_id.nunique()),reference='GPT only, not human truth',
        weather='Regional proxy/historical alignment, not fullG or live data availability',scores=scores,tradeoffs=tradeoffs,all_inputs_unchanged=True)
    S.dump(out/'report.json',report)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>SD photo + weather</title><style>body{font:16px system-ui;margin:30px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}</style><h1>SD GPT-reference: fixed photo + weather comparison</h1><p>Original624 training only. Regional historical ASOS, not fullG/live availability. No SD tuning or final selection.</p>'+pd.DataFrame(scores)[['subset','method','macro_f1','f1_1','recall_2','accuracy','alert_roc_auc','alert_average_precision']].round(5).to_html(index=False)+'<h2>Alert tradeoffs</h2>'+pd.DataFrame(tradeoffs).to_html(index=False),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.data,args.out)
