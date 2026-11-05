import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("conditional_threshold_probe")


@pytest.mark.parametrize("policy",["B_threshold_inner","AB_threshold_inner","recall_guard_inner"])
def test_baseline_feasible_and_recall_guard_respects_both_constraints(module,policy):
    y=np.array([0,0,1,1,2,2]); a=np.array([.2,.6,.4,.8,.9,.7]); b=np.array([.2,.6,.6,.7,.4,.8])
    selected=module.choose_thresholds(y,a,b,policy)
    base=module.decision_stats(y,module.V.decide(a,b))
    assert selected["macro_f1"]>=base["macro_f1"]-1e-12
    assert selected["ta"] in module.GRID and selected["tb"] in module.GRID
    if policy=="B_threshold_inner":assert selected["ta"]==.5
    if policy=="recall_guard_inner":
        assert selected["severe_recall"]>=base["severe_recall"]
        assert selected["extreme_errors"]<=base["extreme_errors"]


def test_thresholds_and_rule_choice_ignore_outer_test_labels(module,monkeypatch):
    train=pd.DataFrame(dict(id=[str(i) for i in range(18)],group=np.repeat(np.arange(6),3),
                            label=np.tile([0,1,2],6),x=np.linspace(.1,.9,18)))
    test=train.assign(id=lambda x:"test"+x.id,group=99)
    aux=train.iloc[:3].assign(id=lambda x:"aux"+x.id,group=77)
    def fit(tr,aux,te):
        assert not set(tr.group)&set(te.group)
        assert not set(aux.group)&set(te.group)
        assert not (set(tr.id)|set(aux.id))&set(te.id)
        return te.x.to_numpy(),1-te.x.to_numpy()
    monkeypatch.setattr(module,"fit_fixed_pair",fit)
    monkeypatch.setattr(module,"inner_rule_predictions",lambda tr,aux,te:{k:np.ones(len(te),int) for k in module.RULES})
    first,pair,choice,traces=module.predict(train,aux,test.drop(columns="label"))
    second,_,other,_=module.predict(train,aux,test.assign(label=2-test.label))
    assert choice==other and choice["chosen_rule"]=="source_rule_tuned"
    assert len(traces)==len(train) and {r["inner_fold"] for r in traces}=={0,1,2}
    for name in first:np.testing.assert_array_equal(first[name],second[name])


def test_A_uses_original_rows_and_B_only_augmented_positive_rows(module,monkeypatch):
    train=pd.DataFrame(dict(id=["s0","s1","s2"],group=["s"]*3,image_sha256=["h0","h1","h2"],label=[0,1,2]))
    aux=pd.DataFrame(dict(id=["a0","a2"],group=["a"]*2,image_sha256=["ha0","ha2"],label=[0,2]))
    test=pd.DataFrame(dict(id=["t"],group=["t"],image_sha256=["ht"]))
    calls=[]
    def binary(tr,te,columns,y,kind):
        calls.append((list(tr.id),list(np.asarray(y,int))))
        return np.full(len(te),.4)
    monkeypatch.setattr(module.V,"binary",binary)
    module.fit_fixed_pair(train,aux,test)
    assert calls==[(["s0","s1","s2"],[0,1,1]),(["s1","s2","a2"],[0,1,1])]


def test_undefined_policy_and_existing_output_rejected(module,tmp_path):
    with pytest.raises(ValueError,match="Unknown"):
        module.choose_thresholds([0],[.2],[.2],"test_selected")
    with pytest.raises(ValueError,match="new private output"):
        module.run(tmp_path,tmp_path)
