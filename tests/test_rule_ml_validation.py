import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("rule_ml_validation")


def frame():
    return pd.DataFrame(dict(id=[str(i) for i in range(18)], group=np.repeat(np.arange(6),3),
                             label=np.tile([0,1,2],6), sun_alt=np.linspace(-5,50,18),
                             gap=np.linspace(0,80,18), cloud=np.tile([0,5,9],6)))


def test_vectorized_rules_exactly_reproduce_original_grids_and_ties(module):
    train = module.geometry(frame())
    test = module.geometry(frame().iloc[::-1].copy())
    actual = module.rule_predictions(train,test)
    for name, fn in [("rule_hand",module.Q.rule_hand),("rule_tuned",module.Q.rule_tuned),("rule_physics",module.Q.rule_physics)]:
        np.testing.assert_array_equal(actual[name],fn(train,test).argmax(axis=1))
    y=np.array([0,0,1]); p=np.array([[0,0,0],[2,1,1]])
    np.testing.assert_allclose(module.macro_matrix(y,p),[module.Q.macro_f1(y,row) for row in p])


def test_conditional_decisions_and_invalid_probabilities(module):
    np.testing.assert_array_equal(module.decide([.49,.5,.8],[1,.49,.5]),[0,1,2])
    with pytest.raises(ValueError,match="finite"):
        module.decide([np.nan],[.5])
    with pytest.raises(ValueError,match="finite"):
        module.decide([.2,1],[.5])


def test_inner_splits_keep_whole_capture_groups(module):
    data=frame()
    seen=np.zeros(len(data),int)
    for a,b in module.inner_splits(data):
        assert not set(data.iloc[a].group)&set(data.iloc[b].group)
        seen[b]+=1
    assert np.all(seen==1)


def test_selection_and_prediction_do_not_read_test_labels(module,monkeypatch):
    train=module.geometry(frame())
    test=module.geometry(frame().assign(id=lambda x:"test"+x.id,group=99))
    calls=[]
    def stub(a,b,kind):
        assert not set(a.id)&set(b.id)
        assert not set(a.group)&set(b.group)
        calls.append((tuple(a.id),tuple(b.id)))
        x=np.clip(b.gap.to_numpy()/100,0,1)
        return x,1-x
    monkeypatch.setattr(module,"fit_pair",stub)
    first,choice,_=module.predict(train,test.drop(columns="label"))
    second,other,_=module.predict(train,test.assign(label=2-test.label))
    assert choice==other and calls
    for key in first:
        np.testing.assert_array_equal(first[key],second[key])


def test_constant_or_empty_binary_does_not_need_a_model(module):
    data=frame()
    np.testing.assert_array_equal(module.binary(data,data,[],np.ones(len(data)),"geometry_cat"),np.ones(len(data)))
    np.testing.assert_array_equal(module.binary(data.iloc[:0],data,[],[],"geometry_cat"),np.zeros(len(data)))


def test_existing_output_rejected_before_input_reads(module,tmp_path):
    with pytest.raises(ValueError,match="new output"):
        module.run(tmp_path,tmp_path)
