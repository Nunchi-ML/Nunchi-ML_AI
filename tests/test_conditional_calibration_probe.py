import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("binary_sigmoid"),importlib.import_module("conditional_calibration_probe")


def test_sigmoid_keeps_order_and_improves_training_logloss(modules):
    from nunchi_ml.glare.evaluation import binary_scores
    s,_=modules;p=np.linspace(.2,.95,30);y=np.array([0]*24+[1]*6)
    model=s.MonotoneSigmoid().fit(p,y);q=model.predict(p)
    assert np.all(np.diff(q)>0) and model.slope_>0
    assert binary_scores(y,q)["log_loss"]<=binary_scores(y,p)["log_loss"]
    assert model.metadata()["n"]==30


def test_single_class_is_explicit_identity_and_invalid_input_rejected(modules):
    s,_=modules;p=np.array([.1,.4,.8]);model=s.MonotoneSigmoid().fit(p,[0,0,0])
    np.testing.assert_array_equal(model.predict(p),p)
    assert model.metadata()["status"]=="identity_single_class"
    with pytest.raises(ValueError,match="finite"):
        s.MonotoneSigmoid().fit([np.nan],[1])


def test_B_calibrator_only_sees_original_positive_OOF_labels(modules,monkeypatch):
    _,m=modules;seen=[]
    class Spy:
        def fit(self,p,y):seen.append((list(p),list(np.asarray(y,int))));return self
        def predict(self,p):return np.asarray(p)
        def metadata(self):return {"status":"spy"}
    monkeypatch.setattr(m.S,"MonotoneSigmoid",Spy)
    train=pd.DataFrame(dict(label=[0,1,2,0]))
    m.apply_policies(train,np.array([.2,.6,.9,.7]),np.array([.9,.2,.8,.3]),np.array([.9]),np.array([.8]))
    assert seen==[([.2,.8],[0,1])]


def test_original_B_inner_fit_excludes_entire_capture(modules,monkeypatch):
    _,m=modules
    train=pd.DataFrame(dict(id=[str(i) for i in range(6)],group=["a","a","b","b","c","c"],label=[1,2,1,2,1,2]))
    trace=train.assign(inner_fold=[0,0,1,1,2,2])
    def binary(tr,te,cols,target,kind):
        assert not set(tr.group)&set(te.group)
        return np.full(len(te),.4)
    monkeypatch.setattr(m.T.V,"binary",binary)
    values,_=m.source_inner_B(train,trace)
    np.testing.assert_allclose(values,.4)
    with pytest.raises(ValueError,match="identity"):
        m.source_inner_B(train,pd.concat([trace,trace.iloc[:1]]))


def test_matched_repeat_counts_do_not_treat_repeats_as_independent_samples(modules):
    _,m=modules;records=[]
    for r in range(30):
        for name,value in [("base",.5),("new",.6 if r<15 else (.4 if r<25 else .5))]:
            records.append(dict(test="group_cv320",repeat=r,method=name,macro_f1=value))
    row=next(x for x in m.paired_repeats(pd.DataFrame(records),"base") if x["method"]=="new")
    assert (row["improved"],row["tied"],row["worse"])==(15,5,10)
    assert row["improved_fraction"]==.5
    assert "independent_ci" not in row


def test_existing_calibration_output_is_preserved(modules,tmp_path):
    _,m=modules
    with pytest.raises(ValueError,match="new private"):
        m.run(tmp_path,tmp_path)
