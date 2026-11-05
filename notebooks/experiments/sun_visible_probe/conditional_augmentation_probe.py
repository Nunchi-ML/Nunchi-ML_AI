"""Augment conditional training with existing AIHub labels; evaluate original320.

Give tuned rule comparators the same augmentation pool. Never score AIHub as a
test in an augmented condition. Targets, outer folds and .5 boundaries stay fixed.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import rule_ml_validation as V

NEW=["aug_rule_tuned","aug_rule_physics","B_only_aug","AB_aug","B_geometry_aug"]
REFERENCES={"source_rule_hand":"rule_hand","source_rule_tuned":"rule_tuned",
            "source_rule_physics":"rule_physics","source_conditional":"conditional_lr"}


def training_pool(train, auxiliary, test):
    pool=pd.concat([train,auxiliary],ignore_index=True)
    if (not pool.id.is_unique or pool.image_sha256.duplicated().any()
            or set(pool.id)&set(test.id) or set(pool.group)&set(test.group) or set(pool.image_sha256)&set(test.image_sha256)):
        raise ValueError("Augmented pool has duplicate or evaluation capture/image leakage")
    return pool


def predict(train, auxiliary, test):
    pool=training_pool(train,auxiliary,test)
    a,b=V.fit_pair(train,test,"conditional_lr")
    positive=pool[pool.label.ge(1)]
    aug_b=V.binary(positive,test,V.M.R.G,positive.label.eq(2),"conditional_lr")
    geo_b=V.binary(positive,test,V.M.R.G+V.EXTRA,positive.label.eq(2),"geometry_lr")
    aug_a=V.binary(pool,test,V.M.R.G+V.M.R.P.REL_R,pool.label.ge(1),"conditional_lr")
    rules=V.rule_predictions(pool,test)
    predictions=dict(aug_rule_tuned=rules["rule_tuned"],aug_rule_physics=rules["rule_physics"],
        B_only_aug=V.decide(a,aug_b),AB_aug=V.decide(aug_a,aug_b),B_geometry_aug=V.decide(a,geo_b))
    return predictions,V.decide(a,b),dict(source_a=a,source_b=b,aug_a=aug_a,aug_b=aug_b,geometry_b=geo_b)


def run(data,out):
    data,out=Path(data).resolve(),Path(out).resolve()
    if out.exists() or out.parent!=data:raise ValueError("Use a new private output")
    prior=data/"rule_ml_validation_20261105";base=data/"road_visibility_20261104/run30"
    files=[prior/"report.json",prior/"predictions.csv",base/"training_features.csv",base/"splits.csv",Path(__file__),Path(V.__file__)]
    hashes={str(p):V.M.sha(p) for p in files};original=json.loads(files[0].read_text(encoding="utf-8"))
    if original["status"]!="complete" or any(V.M.sha(p)!=h for p,h in original["input_sha256"].items()):raise ValueError("Reference provenance mismatch")
    full=V.geometry(pd.read_csv(files[2],dtype={"id":str}));train=full[full.camera.ne("AI Hub")].reset_index(drop=True);aux=full[full.camera.eq("AI Hub")]
    if len(train)!=320 or train.group.nunique()!=24 or len(aux)!=304 or aux.group.nunique()!=3:raise ValueError("Frozen original320/24 and auxiliary304/3 required")
    training_pool(train.iloc[:0],aux,train)
    out.mkdir();specification=dict(status="frozen_before_fit",primary="B_only_aug vs source_rule_tuned and aug_rule_tuned MacroF1 on same original320",
        reason="Original73 severe photos span3captures,71inone; keep A fixed and add existing42AIHubsevere/3captures to positive-only B",
        targets="Existing0/1/2 unchanged; no new labels or neural features",evaluation="Frozen original320/24capture/5folds x30 plus3heldoriginalcameras; NO AIHub test",
        variants=dict(B_only_aug="Original-only A; B positives from outer-original-training+AIHub304",
            AB_aug="Both A/B trained on outer-original-training+AIHub304",B_geometry_aug="Original-only A; augmented B with G+gap/theta/logLv",
            rules="Same augmented pool available to exact train-tuned geometry and physics rules; original-source rule controls retained"),
        hyperparameters="Balanced LR C1, train-only scaler, fixed.5/.5; existing exact rule grids; no tuning/selection after test",
        limitation="Additional-data intervention on reused development benchmark; separate from unchanged-training-pool CV; no independent external proof or final selection",
        input_sha256=hashes)
    (out/"specification_before_run.json").write_text(json.dumps(specification,ensure_ascii=False,indent=2),encoding="utf-8");shutil.copyfile(__file__,out/"source_before_run.py")
    frozen=pd.read_csv(files[1],dtype={"id":str});splits=pd.read_csv(files[3],dtype={"id":str});scores=[];records=[];probabilities=[]
    regimes=[("group_cv320",r,train) for r in range(30)]
    regimes += [("held_camera:"+str(c),-1,train[train.camera.eq(c)]) for c in train.camera.unique()]
    for regime,repeat,test in regimes:
        predictions={name:np.full(len(test),-1,int) for name in NEW};baseline=np.full(len(test),-1,int)
        for name,old_name in REFERENCES.items():
            old=frozen[frozen.test.eq(regime)&frozen["repeat"].eq(repeat)&frozen.method.eq(old_name)]
            if not old.id.is_unique or set(old.id)!=set(test.id):raise ValueError("Reference coverage mismatch")
            old=old.set_index("id").loc[test.id]
            if not np.array_equal(old.label,test.label):raise ValueError("Original label changed")
            predictions[name]=old.pred.to_numpy()
        if repeat>=0:
            part=splits[splits["repeat"].eq(repeat)]
            if len(part)!=320 or not part.id.is_unique or set(part.id)!=set(train.id):raise ValueError("Frozen split mismatch")
            folds=[(f,train[~train.id.isin(part[part.fold.eq(f)].id)],train[train.id.isin(part[part.fold.eq(f)].id)]) for f in range(5)]
        else:folds=[(-1,train[~train.id.isin(test.id)],test)]
        seen=np.zeros(len(test),int)
        for fold,a,b in folds:
            pred,reproduced,probs=predict(a,aux,b);mask=test.id.isin(b.id).to_numpy();seen[mask]+=1;baseline[mask]=reproduced
            for name,p in pred.items():predictions[name][mask]=p
            probabilities.extend(dict(test=regime,repeat=repeat,fold=fold,id=cid,**{k:float(v[i]) for k,v in probs.items()}) for i,cid in enumerate(b.id))
        if not np.all(seen==1) or not np.array_equal(baseline,predictions["source_conditional"]):raise ValueError("Baseline reproduction or OOF coverage failed")
        for name,p in predictions.items():
            scores.append(dict(test=regime,repeat=repeat,method=name,**V.metric(test.label,p)))
            records.extend(dict(test=regime,repeat=repeat,method=name,id=cid,group=g,label=int(y),pred=int(v)) for cid,g,y,v in zip(test.id,test.group,test.label,p))
        pd.DataFrame(scores).to_csv(out/"scores.csv",index=False,encoding="utf-8-sig");print("Conditional augmentation",regime,repeat+1,flush=True)
    frame=pd.DataFrame(scores);summary=frame.groupby(["test","method"],sort=False)[[c for c in frame.select_dtypes("number") if c!="repeat"]].mean().reset_index()
    summary.to_csv(out/"summary.csv",index=False,encoding="utf-8-sig");pd.DataFrame(records).to_csv(out/"predictions.csv",index=False,encoding="utf-8-sig");pd.DataFrame(probabilities).to_csv(out/"binary_probabilities.csv",index=False,encoding="utf-8-sig")
    names=list(REFERENCES)+NEW;groups=sorted(train.group.unique());index={g:i for i,g in enumerate(groups)};counts=np.zeros((24,30,len(names),3,3),int)
    for r in pd.DataFrame(records).query("test == 'group_cv320'").itertuples():counts[index[r.group],r.repeat,names.index(r.method),r.label,r.pred]+=1
    draws=[];rng=np.random.default_rng(0)
    for _ in range(2000):
        c=counts[rng.integers(0,24,24)].sum(axis=0);diag=np.diagonal(c,axis1=-2,axis2=-1);den=c.sum(axis=-2)+c.sum(axis=-1)
        draws.append(np.divide(2*diag,den,out=np.zeros(diag.shape,float),where=den!=0).mean(axis=-1).mean(axis=0))
    draws=np.asarray(draws);intervals=[dict(method=k,baseline=b,low=float(np.quantile(draws[:,i]-draws[:,names.index(b)],.025)),high=float(np.quantile(draws[:,i]-draws[:,names.index(b)],.975))) for b in ["source_rule_tuned","aug_rule_tuned","source_conditional"] for i,k in enumerate(names)]
    pd.DataFrame(intervals).to_csv(out/"paired_capture_intervals.csv",index=False,encoding="utf-8-sig")
    if any(V.M.sha(p)!=h for p,h in (hashes|original["image_sha256"]).items()):raise ValueError("Source/data/image changed")
    result=dict(status="complete",original_photos=320,auxiliary_training_photos=304,auxiliary_severe=42,aihub_used_as_test=False,source_baseline_reproduced=9600+320,limitations=specification["limitation"],input_sha256=hashes)
    (out/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");(out/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>Conditional augmentation vs tuned rules</title><h1>Original320 evaluation / same extra training pool for rules</h1><p>AIHub304 used only as training data. Not an AIHub test or independent external proof.</p>'+summary.round(5).to_html(index=False),encoding="utf-8")
    print(summary[["test","method","macro_f1","f1_1","recall_2"]].round(5).to_string(index=False),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--data",type=Path,default=Path("data"));parser.add_argument("--out",type=Path,required=True);args=parser.parse_args();run(args.data,args.out)
