import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("severity_stage_probe")


def test_both_severity_learners_fit_only_original_positive_rows(module,monkeypatch):
    data=pd.DataFrame({**{c:[1.,2.,3.,4.] for c in module.BASIC},"label":[0,1,2,0]})
    fitted=[]
    class Spy:
        classes_=np.array([0,1])
        def fit(self,x,y):
            fitted.append((list(x.index),list(y)))
            return self
        def predict_proba(self,x):return np.tile([.6,.4],(len(x),1))
    monkeypatch.setattr(module,"DecisionTreeClassifier",lambda **kwargs:Spy())
    monkeypatch.setattr(module,"make_pipeline",lambda *args:Spy())
    for kind in ["tree","quadratic"]:
        np.testing.assert_allclose(module.severity(data,data.drop(columns="label"),kind),.4)
    assert fitted==[([1,2],[0,1]),([1,2],[0,1])]


def test_one_stage_controls_preserve_the_other_stage(module,monkeypatch):
    monkeypatch.setattr(module.V,"fit_pair",lambda tr,te,k: (np.array([.9,.1]),np.array([.1,.9])) if k=="conditional_lr" else (np.array([.1,.9]),np.array([.9,.1])))
    monkeypatch.setattr(module,"severity",lambda tr,te,k:np.array([.8,.8]))
    pred,baseline,pairs=module.predict(None,None)
    np.testing.assert_array_equal(baseline,[1,0])
    np.testing.assert_array_equal(pred["A_geometry_only"],[0,2])
    np.testing.assert_array_equal(pred["B_geometry_only"],[2,0])
    for name in ["B_geometry_only","B_tree_only","B_quadratic_only"]:
        np.testing.assert_array_equal(pairs[name][0],[.9,.1])


def test_empty_and_single_class_positive_sets_are_explicit(module):
    data=pd.DataFrame({**{c:[1.,2.] for c in module.BASIC},"label":[0,0]})
    np.testing.assert_array_equal(module.severity(data,data,"tree"),[0,0])
    np.testing.assert_array_equal(module.severity(data.assign(label=2),data,"quadratic"),[1,1])


def test_existing_output_rejected_before_reading_sources(module,tmp_path):
    with pytest.raises(ValueError,match="new private output"):
        module.run(tmp_path,tmp_path)
