"""Three-grade evaluation with optional saved probabilities, never refitting.

Macro F1 measures the supplied final decisions; AUC/AP measure score ranking.
Hard decisions are never silently converted to probabilities. Multiclass Brier
is mean(sum((p-onehot(y))**2)), range [0,2], without division by class count.
"""
import json

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             cohen_kappa_score, confusion_matrix, log_loss,
                             mean_absolute_error, precision_recall_fscore_support,
                             roc_auc_score)


def binary_scores(y, probability):
    y, probability = np.asarray(y, int), np.asarray(probability, float)
    if y.ndim != 1 or y.shape != probability.shape or not len(y) or not np.isin(y,[0,1]).all():
        raise ValueError("Matched nonempty binary targets required")
    if not np.isfinite(probability).all() or ((probability < -1e-12)|(probability > 1+1e-12)).any():
        raise ValueError("Valid binary probabilities required")
    probability=np.clip(probability,0,1)
    both=len(np.unique(y))==2
    return dict(roc_auc=float(roc_auc_score(y,probability)) if both else None,
                average_precision=float(average_precision_score(y,probability)) if both else None,
                brier=float(np.mean((probability-y)**2)),
                log_loss=float(log_loss(y,np.column_stack([1-probability,probability]),labels=[0,1])),
                positive_support=int(y.sum()),negative_support=int((y==0).sum()),ranking_defined=both)


def evaluate_grades(y,pred,probabilities=None,conditional_severe=None):
    y,pred=np.asarray(y),np.asarray(pred)
    if y.ndim!=1 or y.shape!=pred.shape or not len(y) or not np.isin(y,[0,1,2]).all() or not np.isin(pred,[0,1,2]).all():
        raise ValueError("Matched nonempty original grades0/1/2 required")
    y,pred=y.astype(int),pred.astype(int)
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    qwk=float(cohen_kappa_score(y,pred,labels=[0,1,2],weights="quadratic")) if len(np.unique(np.r_[y,pred]))>1 else None
    result=dict(n=len(y),macro_f1=float(f1.mean()),weighted_f1=float(np.average(f1,weights=support)),
        accuracy=float(accuracy_score(y,pred)),qwk=qwk,mae=float(mean_absolute_error(y,pred)),
        count_0_to_2=int(((y==0)&(pred==2)).sum()),count_2_to_0=int(((y==2)&(pred==0)).sum()),
        confusion_matrix=json.dumps(confusion_matrix(y,pred,labels=[0,1,2]).tolist()),
        probability_status="not_saved" if probabilities is None else "provided_not_assumed_calibrated",
        macro_ovr_auc=None,macro_average_precision=None,multiclass_log_loss=None,multiclass_brier=None)
    for c in range(3):
        result.update({f"precision_{c}":float(precision[c]),f"recall_{c}":float(recall[c]),
                       f"f1_{c}":float(f1[c]),f"support_{c}":int(support[c])})
    if probabilities is None:
        if conditional_severe is not None:raise ValueError("Joint probabilities required with conditional scores")
        return result
    probability=np.asarray(probabilities,float)
    if (probability.shape!=(len(y),3) or not np.isfinite(probability).all()
            or ((probability < -1e-12)|(probability > 1+1e-12)).any() or not np.allclose(probability.sum(axis=1),1,atol=1e-6,rtol=0)):
        raise ValueError("Finite normalized three-class probabilities required")
    original_probability=probability.copy()
    result["probability_row_sum_error_max"]=float(np.max(abs(probability.sum(axis=1)-1)))
    # Only numerical roundoff is accepted: typical float32 row sums differ by
    # ~1e-7 and old composition can contain -2e-16. Record the correction size.
    probability=np.clip(probability,0,1)
    probability=probability/probability.sum(axis=1,keepdims=True)
    result["probability_roundoff_max"]=float(np.max(abs(probability-original_probability)))
    result["multiclass_log_loss"]=float(log_loss(y,probability,labels=[0,1,2]))
    result["multiclass_brier"]=float(np.mean(np.sum((probability-np.eye(3)[y])**2,axis=1)))
    aucs=[];aps=[]
    for c in range(3):
        scores=binary_scores(y==c,probability[:,c]);aucs.append(scores["roc_auc"]);aps.append(scores["average_precision"])
        result.update({f"class_{c}_{k}":v for k,v in scores.items()})
    if all(v is not None for v in aucs):result["macro_ovr_auc"]=float(np.mean(aucs))
    if all(v is not None for v in aps):result["macro_average_precision"]=float(np.mean(aps))
    alert=probability[:,1:].sum(axis=1)
    for name,truth,values in [("alert",y>=1,alert),("severe",y==2,probability[:,2])]:
        result.update({name+"_"+k:v for k,v in binary_scores(truth,values).items()})
    positive=y>=1
    if conditional_severe is None:
        conditional=np.divide(probability[:,2],alert,out=np.full(len(y),np.nan),where=alert>0)
        result["conditional_score_source"]="derived_from_joint"
    else:
        conditional=np.asarray(conditional_severe,float)
        if conditional.shape!=(len(y),) or not np.isfinite(conditional).all() or ((conditional<0)|(conditional>1)).any():
            raise ValueError("Valid matched conditional probabilities required")
        if not np.allclose(alert*conditional,probability[:,2],atol=1e-8,rtol=0):
            raise ValueError("Conditional and joint probabilities disagree")
        result["conditional_score_source"]="saved_binary_B"
    result["conditional_severe_support"]=int(positive.sum())
    result["conditional_severe_coverage"]=int(np.isfinite(conditional[positive]).sum())
    result["conditional_severe_roc_auc"]=None
    result["conditional_severe_average_precision"]=None
    if positive.any() and np.isfinite(conditional[positive]).all():
        result.update({"conditional_severe_"+k:v for k,v in binary_scores(y[positive]==2,conditional[positive]).items()})
    return result
