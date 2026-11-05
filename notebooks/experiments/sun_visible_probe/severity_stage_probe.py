"""One-stage ablations following the failed combined geometry/CatBoost trial.

Keep original targets, saved outer splits and .5/.5 boundaries. This is another
development comparison, not a new independent test or a population probability.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import rule_ml_validation as V

NEW = ["A_geometry_only", "B_geometry_only", "B_tree_only", "B_quadratic_only"]
REFERENCES = ["rule_hand", "rule_tuned", "rule_physics", "conditional_lr"]
BASIC = ["sun_alt", "gap", "cloud", "sunshine"]


def severity(train, test, kind):
    positive = train[train.label.ge(1)]
    y = positive.label.eq(2).to_numpy(int)
    if len(np.unique(y)) < 2:
        return np.full(len(test), float(y[0]) if len(y) else 0.)
    if kind == "tree":
        model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1, class_weight="balanced", random_state=0)
    elif kind == "quadratic":
        model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False),
                              LogisticRegression(C=.1, class_weight="balanced", max_iter=3000))
    else:
        raise ValueError("Unknown fixed severity learner")
    model.fit(positive[BASIC], y)
    return model.predict_proba(test[BASIC])[:, list(model.classes_).index(1)]


def predict(train, test):
    a, b = V.fit_pair(train, test, "conditional_lr")
    geo_a, geo_b = V.fit_pair(train, test, "geometry_lr")
    pairs = dict(A_geometry_only=(geo_a,b), B_geometry_only=(a,geo_b),
                 B_tree_only=(a,severity(train,test,"tree")),
                 B_quadratic_only=(a,severity(train,test,"quadratic")))
    return {name:V.decide(*pair) for name,pair in pairs.items()}, V.decide(a,b), pairs


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or out.parent != data:
        raise ValueError("Use a new private output")
    prior=data/"rule_ml_validation_20261105"; base=data/"road_visibility_20261104/run30"
    files=[prior/"report.json",prior/"predictions.csv",base/"training_features.csv",base/"splits.csv",
           Path(__file__),Path(V.__file__)]
    hashes={str(p):V.M.sha(p) for p in files}
    report=json.loads(files[0].read_text(encoding="utf-8"))
    if report["status"]!="complete" or any(V.M.sha(p)!=h for p,h in report["input_sha256"].items()):
        raise ValueError("Completed unchanged first comparison required")
    full=V.geometry(pd.read_csv(files[2],dtype={"id":str})); train=full[full.camera.ne("AI Hub")].reset_index(drop=True)
    if len(train)!=320 or train.group.nunique()!=24 or not full.id.is_unique:
        raise ValueError("Original320/24groups required")
    frozen=pd.read_csv(files[1],dtype={"id":str}); splits=pd.read_csv(files[3],dtype={"id":str})
    out.mkdir()
    specification=dict(status="frozen_before_fit",primary="B_geometry_only vs rule_tuned and conditional_lr MacroF1",
        motivation="Combined geometry LR fell .63568->.60817; isolate A/B changes, keep successful A for severe-stage alternatives. Cat severe recall0, no more Cat search.",
        new=NEW, references=REFERENCES, original="320photos/24captures, frozen5folds x30, original0/1/2; fixed.5/.5",
        variants=dict(A_geometry_only="Change A features only; original G-only LR B",
            B_geometry_only="Original G+relative12 LR A; LR B on G+gap/theta/log-Lv",
            B_tree_only="Original A; positive-only balanced depth2/minleaf1 tree B on sun_alt/gap/cloud/sunshine",
            B_quadratic_only="Original A; positive-only standardize->degree2 no-bias polynomial->balanced LR C.1 B on same4geometryfeatures"),
        selection="No model/threshold tuning; all four predefined ablations reported. Highest observed candidate is not an independently selected winner.",
        followups="Fixed held3cameras/reusedAIHub304, no retuning from outcomes",
        limits="Sequential development experiment on reused data; not independent proof. FrozenOOF capture composition intervals, no refitting.", input_sha256=hashes)
    (out/"specification_before_run.json").write_text(json.dumps(specification,ensure_ascii=False,indent=2),encoding="utf-8")
    shutil.copyfile(__file__,out/"source_before_run.py")
    records=[];scores=[];probabilities=[]
    regimes=[("group_cv320",r,train) for r in range(30)]
    regimes += [("reused_AIHub",-1,full[full.camera.eq("AI Hub")])]
    regimes += [("held_camera:"+str(c),-1,train[train.camera.eq(c)]) for c in train.camera.unique()]
    for regime,repeat,test in regimes:
        reference={}
        for name in REFERENCES:
            old=frozen[frozen.test.eq(regime)&frozen["repeat"].eq(repeat)&frozen.method.eq(name)]
            if not old.id.is_unique or set(old.id)!=set(test.id):raise ValueError("Frozen reference coverage")
            old=old.set_index("id").loc[test.id]
            if not np.array_equal(old.label.to_numpy(),test.label.to_numpy()):raise ValueError("Frozen label mismatch")
            reference[name]=old.pred.to_numpy()
        predictions={k:np.full(len(test),-1,int) for k in NEW}; reproduced=np.full(len(test),-1,int)
        if repeat>=0:
            part=splits[splits["repeat"].eq(repeat)]
            if len(part)!=320 or not part.id.is_unique or set(part.id)!=set(train.id):raise ValueError("Split mismatch")
            folds=[(f,train[~train.id.isin(part[part.fold.eq(f)].id)],train[train.id.isin(part[part.fold.eq(f)].id)]) for f in range(5)]
        else:
            a=train if regime=="reused_AIHub" else train[~train.id.isin(test.id)]
            folds=[(-1,a,test)]
        seen=np.zeros(len(test),int)
        for fold,a,b in folds:
            if set(a.group)&set(b.group):raise ValueError("Capture leakage")
            mask=test.id.isin(b.id).to_numpy()
            pred,baseline,pairs=predict(a,b)
            reproduced[mask]=baseline;seen[mask]+=1
            for name,p in pred.items():
                predictions[name][mask]=p
                probabilities.extend(dict(test=regime,repeat=repeat,fold=fold,method=name,id=cid,a=float(pa),b=float(pb)) for cid,pa,pb in zip(b.id,*pairs[name]))
        if not np.all(seen==1) or not np.array_equal(reproduced,reference["conditional_lr"]):raise ValueError("Conditional baseline reproduction failed")
        for name,p in (reference|predictions).items():
            scores.append(dict(test=regime,repeat=repeat,method=name,**V.metric(test.label,p)))
            records.extend(dict(test=regime,repeat=repeat,method=name,id=cid,group=g,label=int(y),pred=int(value)) for cid,g,y,value in zip(test.id,test.group,test.label,p))
        pd.DataFrame(scores).to_csv(out/"scores.csv",index=False,encoding="utf-8-sig")
        print("Severity-stage fixed ablation",regime,repeat+1,flush=True)
    frame=pd.DataFrame(scores);summary=frame.groupby(["test","method"],sort=False)[[c for c in frame.select_dtypes("number") if c!="repeat"]].mean().reset_index()
    summary.to_csv(out/"summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(out/"predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(probabilities).to_csv(out/"binary_probabilities.csv",index=False,encoding="utf-8-sig")
    names=REFERENCES+NEW;groups=sorted(train.group.unique());lookup={g:i for i,g in enumerate(groups)}
    counts=np.zeros((24,30,len(names),3,3),int)
    for row in pd.DataFrame(records).query("test == 'group_cv320'").itertuples():counts[lookup[row.group],row.repeat,names.index(row.method),row.label,row.pred]+=1
    rng=np.random.default_rng(0);draws=[]
    for _ in range(2000):
        c=counts[rng.integers(0,24,24)].sum(axis=0);diag=np.diagonal(c,axis1=-2,axis2=-1);den=c.sum(axis=-2)+c.sum(axis=-1)
        draws.append(np.divide(2*diag,den,out=np.zeros(diag.shape,float),where=den!=0).mean(axis=-1).mean(axis=0))
    draws=np.asarray(draws)
    intervals=[dict(method=k,baseline=b,low=float(np.quantile(draws[:,i]-draws[:,names.index(b)],.025)),high=float(np.quantile(draws[:,i]-draws[:,names.index(b)],.975))) for b in ["rule_tuned","conditional_lr"] for i,k in enumerate(names)]
    pd.DataFrame(intervals).to_csv(out/"paired_capture_intervals.csv",index=False,encoding="utf-8-sig")
    if any(V.M.sha(p)!=h for p,h in (hashes|report["image_sha256"]).items()):raise ValueError("Inputs/images changed")
    result=dict(status="complete",baseline_reproduced=9600+624,limitations=specification["limits"],input_sha256=hashes)
    (out/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>Severity-stage ablation</title><h1>Fixed original320 grouped CV: A/B one-stage changes</h1><p>Reused development data. All predefined candidates retained; no final-model choice.</p>'+summary.round(5).to_html(index=False),encoding="utf-8")
    print(summary[["test","method","macro_f1","f1_1","recall_2"]].round(5).to_string(index=False),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--data",type=Path,default=Path("data"));parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();run(args.data,args.out)
