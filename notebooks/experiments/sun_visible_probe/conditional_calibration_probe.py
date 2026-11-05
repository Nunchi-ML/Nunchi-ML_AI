"""Matched old/new thresholds and OOF B calibration; no outer-test tuning.

Threshold policies use raw OOF. Calibration is evaluated at fixed .5/.5 only;
never fit a calibrator and tune thresholds on those same calibrated labels.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import conditional_threshold_probe as T
import binary_sigmoid as S
from nunchi_ml.glare.evaluation import evaluate_grades

VARIANTS=["fixed","B_threshold","AB_threshold","recall_guard","sigmoid"]
NAMES=[f"{source}_{name}" for source in ["source","aug"] for name in VARIANTS]+["source_rule_tuned","aug_rule_tuned","rule_inner_selected"]


def source_inner_B(train, trace):
    if trace.id.duplicated().any() or set(trace.id)!=set(train.id):raise ValueError("Inner trace identity mismatch")
    trace=trace.set_index("id").loc[train.id]
    if not np.array_equal(trace.label.to_numpy(),train.label.to_numpy()):raise ValueError("Inner labels mismatch")
    result=np.full(len(train),np.nan)
    for fold in sorted(trace.inner_fold.unique()):
        mask=trace.inner_fold.eq(fold).to_numpy();a,b=train[~mask],train[mask]
        if set(a.group)&set(b.group):raise ValueError("Calibration OOF capture leakage")
        positive=a[a.label.ge(1)]
        result[mask]=T.V.binary(positive,b,T.V.M.R.G,positive.label.eq(2),"conditional_lr")
    if not np.isfinite(result).all():raise ValueError("Incomplete original-B OOF")
    return result,trace


def apply_policies(train, oof_a, oof_b, outer_a, outer_b):
    choices=[T.choose_thresholds(train.label.to_numpy(),oof_a,oof_b,p) for p in T.POLICIES]
    predictions={"fixed":T.V.decide(outer_a,outer_b)}
    for name,choice in zip(["B_threshold","AB_threshold","recall_guard"],choices):
        predictions[name]=T.V.decide(outer_a,outer_b,choice["ta"],choice["tb"])
    positive=train.label.ge(1).to_numpy()
    calibrator=S.MonotoneSigmoid().fit(oof_b[positive],train.label.to_numpy()[positive]==2)
    calibrated=calibrator.predict(outer_b)
    predictions["sigmoid"]=T.V.decide(outer_a,calibrated)
    return predictions,calibrated,dict(thresholds=choices,calibrator=calibrator.metadata())


def paired_repeats(scores, baseline):
    pivot=scores[scores.test.eq("group_cv320")].pivot(index="repeat",columns="method",values="macro_f1")
    if len(pivot)!=30 or pivot.isna().any().any():raise ValueError("Complete matched30 repeats required")
    result=[]
    for name in pivot:
        delta=(pivot[name]-pivot[baseline]).to_numpy()
        result.append(dict(method=name,baseline=baseline,n=30,mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),
                           improved=int((delta>1e-12).sum()),tied=int((abs(delta)<=1e-12).sum()),worse=int((delta < -1e-12).sum()),
                           improved_fraction=float((delta>1e-12).mean()),min_delta=float(delta.min()),max_delta=float(delta.max())))
    return result


def run(data,out):
    data,out=Path(data).resolve(),Path(out).resolve()
    if out.exists() or out.parent!=data:raise ValueError("Use a new private calibration output")
    prior=data/"conditional_threshold_20261105";aug=data/"conditional_augmentation_20261105"
    files=[prior/"report.json",prior/"inner_oof.csv",prior/"predictions.csv",aug/"binary_probabilities.csv",
        data/"road_visibility_20261104/run30/training_features.csv",data/"road_visibility_20261104/run30/splits.csv",
        Path(__file__),Path(S.__file__),Path(T.__file__),Path(T.E.__file__)]
    hashes={str(p):T.V.M.sha(p) for p in files};previous=json.loads(files[0].read_text(encoding="utf-8"))
    if previous["status"]!="complete" or any(T.V.M.sha(p)!=h for p,h in previous["input_sha256"].items()):raise ValueError("Previous fixed-model provenance changed")
    full=T.V.geometry(pd.read_csv(files[4],dtype={"id":str}));original=full[full.camera.ne("AI Hub")].reset_index(drop=True)
    if len(original)!=320 or original.group.nunique()!=24:raise ValueError("Original320/24 required")
    traces=pd.read_csv(files[1],dtype={"id":str});frozen=pd.read_csv(files[2],dtype={"id":str});raw=pd.read_csv(files[3],dtype={"id":str});splits=pd.read_csv(files[5],dtype={"id":str})
    out.mkdir();spec=dict(status="frozen_before_fit",main_objective="Original-grade MacroF1 vs strong rules; no change of target",
        calibration_primary="aug_sigmoid vs aug_fixed: conditional-B and joint Logloss/Brier, AUC/AP retention; F1/severe errors always reported",
        matched_thresholds="Old and augmented models both receive identical3group OOF/grid/tie rules for B-only,AB,andrecallguard",
        calibration="Unweighted monotone sigmoid on original outer-training POSITIVE cross-fitted B scores only; logit clip1e-12; slope.01..10/intercept-15..15/L2.01; fixed.5/.5",
        separation="No outer320 labels used in fitting/tuning. No combined calibration+threshold tuning on reused calibration labels. No AIHub validation score.",
        reuse="Reuse frozen outer9920 probabilities and augmented inner39040 scores; fit only missing original-B OOF3x153 and306smallcalibrators",
        uncertainty="Matched repeat counts are descriptive, not30independent trials. FixedOOF capture-cluster2000intervals conditional on saved predictions, not algorithm refit confidence.",input_sha256=hashes)
    (out/"specification_before_run.json").write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding="utf-8")
    shutil.copyfile(__file__,out/"runner_before_run.py");shutil.copyfile(S.__file__,out/"calibrator_before_run.py")
    records=[];scores=[];choices=[];new_oof=[]
    regimes=[("group_cv320",r,original) for r in range(30)]+[("held_camera:"+str(c),-1,original[original.camera.eq(c)]) for c in original.camera.unique()]
    for regime,repeat,test in regimes:
        ref=frozen[frozen.test.eq(regime)&frozen["repeat"].eq(repeat)]
        outer=raw[raw.test.eq(regime)&raw["repeat"].eq(repeat)].set_index("id").loc[test.id]
        pred={name:np.full(len(test),-1,int) for name in NAMES};cal={k:np.full(len(test),np.nan) for k in ["source","aug"]}
        for name in ["source_rule_tuned","aug_rule_tuned","rule_inner_selected"]:pred[name]=ref[ref.method.eq(name)].set_index("id").loc[test.id].pred.to_numpy()
        if repeat>=0:
            part=splits[splits["repeat"].eq(repeat)]
            if len(part)!=320 or not part.id.is_unique or set(part.id)!=set(original.id):raise ValueError("Frozen split mismatch")
            folds=[(f,original[~original.id.isin(part[part.fold.eq(f)].id)],original[original.id.isin(part[part.fold.eq(f)].id)]) for f in range(5)]
        else:folds=[(-1,original[~original.id.isin(test.id)],test)]
        seen=np.zeros(len(test),int)
        for fold,train,valid in folds:
            if set(train.group)&set(valid.group):raise ValueError("Outer capture leakage")
            inner=traces[traces.test.eq(regime)&traces["repeat"].eq(repeat)&traces.fold.eq(fold)]
            old_b,inner=source_inner_B(train,inner)
            mask=test.id.isin(valid.id).to_numpy();seen[mask]+=1
            a=outer.loc[valid.id,"source_a"].to_numpy()
            for source,ib,ob in [("source",old_b,outer.loc[valid.id,"source_b"].to_numpy()),("aug",inner.b.to_numpy(),outer.loc[valid.id,"geometry_b"].to_numpy())]:
                values,cb,info=apply_policies(train,inner.a.to_numpy(),ib,a,ob)
                for name,v in values.items():pred[source+"_"+name][mask]=v
                cal[source][mask]=cb;choices.append(dict(test=regime,repeat=repeat,fold=fold,source=source,**info))
            new_oof.extend(dict(test=regime,repeat=repeat,fold=fold,id=cid,label=int(y),a=float(a),source_b=float(b),aug_b=float(c)) for cid,y,a,b,c in zip(train.id,train.label,inner.a,old_b,inner.b))
        if not np.all(seen==1):raise ValueError("Evaluation coverage mismatch")
        for name,p in pred.items():
            source=name.split("_")[0]
            if name not in ["source_rule_tuned","aug_rule_tuned","rule_inner_selected"]:
                a=outer.source_a.to_numpy();b=cal[source] if name.endswith("sigmoid") else outer["source_b" if source=="source" else "geometry_b"].to_numpy()
                probability=np.column_stack([1-a,a*(1-b),a*b])
            else:a=b=probability=None
            scores.append(dict(test=regime,repeat=repeat,method=name,**evaluate_grades(test.label,p,probability,b)))
            records.extend(dict(test=regime,repeat=repeat,method=name,id=cid,group=g,label=int(y),pred=int(v),a=float(a[i]) if a is not None else None,b=float(b[i]) if b is not None else None,**{f"p{k}":float(probability[i,k]) if probability is not None else None for k in range(3)}) for i,(cid,g,y,v) in enumerate(zip(test.id,test.group,test.label,p)))
        for source,old in [("source_fixed","source_conditional"),("aug_fixed","B_geometry_aug"),("aug_B_threshold","B_threshold_inner"),("aug_AB_threshold","AB_threshold_inner"),("aug_recall_guard","recall_guard_inner")]:
            if not np.array_equal(pred[source],ref[ref.method.eq(old)].set_index("id").loc[test.id].pred):raise ValueError("Previous threshold/fixed prediction reproduction failed")
        pd.DataFrame(scores).to_csv(out/"scores.csv",index=False,encoding="utf-8-sig");print("Matched thresholds and B calibration",regime,repeat+1,flush=True)
    frame=pd.DataFrame(scores);frame.groupby(["test","method"],sort=False)[[c for c in frame.select_dtypes("number") if c!="repeat"]].mean().reset_index().to_csv(out/"summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(out/"predictions.csv",index=False,encoding="utf-8-sig");pd.DataFrame(new_oof).to_csv(out/"inner_oof.csv",index=False,encoding="utf-8-sig")
    (out/"calibration_and_thresholds.json").write_text(json.dumps(choices,ensure_ascii=False,indent=2),encoding="utf-8")
    paired=[r for baseline in ["source_fixed","aug_fixed","source_rule_tuned","aug_rule_tuned"] for r in paired_repeats(frame,baseline)]
    pd.DataFrame(paired).to_csv(out/"paired_repeat_deltas.csv",index=False,encoding="utf-8-sig")
    groups=sorted(original.group.unique());gi={g:i for i,g in enumerate(groups)};counts=np.zeros((24,30,len(NAMES),3,3),int)
    for r in pd.DataFrame(records).query("test == 'group_cv320'").itertuples():counts[gi[r.group],r.repeat,NAMES.index(r.method),r.label,r.pred]+=1
    draws=[];rng=np.random.default_rng(0)
    for _ in range(2000):
        c=counts[rng.integers(0,24,24)].sum(axis=0);diag=np.diagonal(c,axis1=-2,axis2=-1);den=c.sum(axis=-2)+c.sum(axis=-1)
        draws.append(np.divide(2*diag,den,out=np.zeros(diag.shape,float),where=den!=0).mean(axis=-1).mean(axis=0))
    draws=np.asarray(draws);intervals=[dict(method=k,baseline=b,low=float(np.quantile(draws[:,i]-draws[:,NAMES.index(b)],.025)),high=float(np.quantile(draws[:,i]-draws[:,NAMES.index(b)],.975))) for b in ["source_fixed","aug_fixed","source_rule_tuned","aug_rule_tuned"] for i,k in enumerate(NAMES)]
    pd.DataFrame(intervals).to_csv(out/"paired_capture_intervals.csv",index=False,encoding="utf-8-sig")
    if any(T.V.M.sha(p)!=h for p,h in (hashes|previous["image_sha256"]).items()):raise ValueError("Frozen sources/inputs/images changed")
    result=dict(status="complete",limits=spec["uncertainty"],input_sha256=hashes,reproduced_fixed_and_aug_threshold_predictions=5*9920)
    (out/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(pd.read_csv(out/"summary.csv").query("test=='group_cv320'")[["method","macro_f1","recall_2","conditional_severe_roc_auc","conditional_severe_log_loss","multiclass_log_loss"]].round(6).to_string(index=False),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--data",type=Path,default=Path("data"));parser.add_argument("--out",type=Path,required=True);args=parser.parse_args();run(args.data,args.out)
