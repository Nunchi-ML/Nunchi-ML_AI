"""Add shared metrics from saved predictions/probabilities, with no model fit.

Original experiment files and source fingerprints remain unchanged. Incomplete
probability coverage is explicit; no AUC is fabricated from hard grade labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from nunchi_ml.glare import evaluation as E

KEY=["test","repeat","method"]
REQUIRED={"id","test","repeat","method","label","pred"}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def with_pair(frame,a,b):
    result=frame.copy()
    result["p0"],result["p1"],result["p2"]=1-a,a*(1-b),a*b
    result["saved_B"]=b
    return result


def join_pair(predictions,probabilities):
    if predictions.duplicated(KEY+["id"]).any() or probabilities.duplicated(KEY+["id"]).any():
        raise ValueError("Duplicate saved prediction identity")
    result=predictions.merge(probabilities,on=KEY+["id"],how="left",validate="one_to_one")
    return result


def audit(data,out):
    data,out=Path(data).resolve(),Path(out).resolve()
    if out.exists() or out.parent!=data:raise ValueError("Use a new private metric audit output")
    hashes={};groups={};inventory=[]
    def read(path):
        hashes[str(path)]=sha(path)
        return pd.read_csv(path,dtype={"id":str})
    def add(source,frame,replace=False):
        for keys,part in frame.groupby(KEY,sort=False):
            key=(source,*keys)
            if key in groups and not replace:raise ValueError("Duplicate evaluation group")
            groups[key]=part
    # Bounded to saved prediction tables; omit known smoke/debug preparations.
    for path in sorted(data.rglob("predictions.csv")):
        if any("smoke" in p.lower() for p in path.relative_to(data).parts):continue
        frame=read(path);supported=REQUIRED.issubset(frame.columns)
        inventory.append(dict(source=path.parent.relative_to(data).as_posix(),rows=len(frame),
                              canonical_long_schema=supported,class_probability_columns={"p0","p1","p2"}.issubset(frame.columns)))
        if supported:add(path.parent.relative_to(data).as_posix(),frame)
    # Latest three comparisons deliberately froze model sources before this
    # posthoc metric request; enrich only this sidecar, never those source files.
    stage="severity_stage_20261105";stage_pred=read(data/stage/"predictions.csv")
    raw=read(data/stage/"binary_probabilities.csv")
    part=join_pair(stage_pred[stage_pred.method.isin(raw.method.unique())],raw)
    if part[["a","b"]].isna().any().any():raise ValueError("Incomplete severity-stage probabilities")
    add(stage,with_pair(part,part.a,part.b),replace=True)
    geometry={}
    for name,column in [("A_geometry_only","a"),("B_geometry_only","b")]:
        geometry[column]=part[part.method.eq(name)][["test","repeat","id",column]]
    combined=geometry["a"].merge(geometry["b"],on=["test","repeat","id"],validate="one_to_one")
    original="rule_ml_validation_20261105"
    original_pred=read(data/original/"predictions.csv")
    restored=original_pred[original_pred.method.eq("geometry_lr")].merge(combined,on=["test","repeat","id"],how="left",validate="one_to_one")
    if restored[["a","b"]].isna().any().any():raise ValueError("Incomplete geometry restoration")
    add(original,with_pair(restored,restored.a,restored.b),replace=True)
    # The unchanged original conditional A/B can also be recovered from the
    # controls, with every row/ID checked instead of using selected subsets.
    a=part[part.method.eq("B_geometry_only")][["test","repeat","id","a"]]
    b=part[part.method.eq("A_geometry_only")][["test","repeat","id","b"]]
    base=a.merge(b,on=["test","repeat","id"],validate="one_to_one")
    for source,predictions in [(original,original_pred),(stage,stage_pred)]:
        joined=predictions[predictions.method.eq("conditional_lr")].merge(base,on=["test","repeat","id"],how="left",validate="one_to_one")
        if joined[["a","b"]].isna().any().any():raise ValueError("Incomplete baseline restoration")
        add(source,with_pair(joined,joined.a,joined.b),replace=True)
    augmented="conditional_augmentation_20261105"
    pred=read(data/augmented/"predictions.csv");raw=read(data/augmented/"binary_probabilities.csv")
    for method,a,b in [("source_conditional","source_a","source_b"),("B_only_aug","source_a","aug_b"),
                       ("AB_aug","aug_a","aug_b"),("B_geometry_aug","source_a","geometry_b")]:
        joined=pred[pred.method.eq(method)].merge(raw,on=["test","repeat","id"],how="left",validate="one_to_one")
        if joined[[a,b]].isna().any().any():raise ValueError("Incomplete augmentation probabilities")
        add(augmented,with_pair(joined,joined[a],joined[b]),replace=True)
    out.mkdir();records=[];coverage=[]
    for index,((source,test,repeat,method),part) in enumerate(groups.items(),1):
        if part.id.duplicated().any():raise ValueError("Duplicate photo within evaluation")
        available={"p0","p1","p2"}.issubset(part.columns) and part[["p0","p1","p2"]].notna().all().all()
        probability=part[["p0","p1","p2"]].to_numpy(float) if available else None
        b=part.saved_B.to_numpy(float) if "saved_B" in part and available else None
        result=E.evaluate_grades(part.label.to_numpy(),part.pred.to_numpy(),probability,b)
        records.append(dict(source=source,test=test,repeat=int(repeat),method=method,**result))
        coverage.append(dict(source=source,test=test,repeat=int(repeat),method=method,n=len(part),
                             probability_available=available,conditional_source=result.get("conditional_score_source"),
                             reason="saved_or_reconstructed_from_exact_stage_controls" if available else "raw_probabilities_not_saved; no_hard_label_AUC"))
        if index%200==0:print("Saved metric groups evaluated",index,"/",len(groups),flush=True)
    frame=pd.DataFrame(records);frame.to_csv(out/"metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(out/"coverage.csv",index=False,encoding="utf-8-sig")
    numeric=[c for c in frame.select_dtypes("number") if c!="repeat"]
    frame.groupby(["source","test","method"],sort=False)[numeric].mean().reset_index().to_csv(out/"summary.csv",index=False,encoding="utf-8-sig")
    hashes[str(Path(__file__))]=sha(__file__);hashes[str(Path(E.__file__))]=sha(E.__file__)
    if any(sha(p)!=h for p,h in hashes.items()):raise ValueError("Saved inputs changed during evaluation")
    shutil.copyfile(__file__,out/"runner_at_audit.py");shutil.copyfile(E.__file__,out/"evaluation_at_audit.py")
    report=dict(status="complete_no_retraining",groups=len(groups),with_probability=int(sum(r["probability_available"] for r in coverage)),
        inventory=inventory,input_sha256=hashes,primary="Original experiment MacroF1 unchanged; AUC/AP/calibration added posthoc, never used to select thresholds",
        limits="Saved cross-fitted scores are development evidence; ranking and calibration differ; AP is average precision, not trapezoidal PR area. Raw probabilities do not imply calibration. Missing classes have undefined AUC/AP, never0.",
        future="Call nunchi_ml.glare.evaluation.evaluate_grades with original final predictions, joint probabilities and actual conditional B when available; save all outputs.")
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ["status","groups","with_probability"]}),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--data",type=Path,default=Path("data"));parser.add_argument("--out",type=Path,required=True);args=parser.parse_args();audit(args.data,args.out)
