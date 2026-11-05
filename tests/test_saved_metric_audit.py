import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("saved_metric_audit")


def test_probability_join_rejects_duplicates_and_keeps_missing_rows(module):
    pred=pd.DataFrame(dict(test=["cv","cv"],repeat=[0,0],method=["m","m"],id=["a","b"],label=[0,2],pred=[0,1]))
    probability=pred.iloc[:1][module.KEY+["id"]].assign(a=.2,b=.4)
    result=module.join_pair(pred,probability)
    assert list(result.id)==["a","b"] and pd.isna(result.loc[1,"a"])
    with pytest.raises(ValueError,match="Duplicate"):
        module.join_pair(pred,pd.concat([probability,probability]))


def test_saved_final_decisions_are_preserved_when_attaching_probabilities(module):
    pred=pd.DataFrame(dict(pred=[2,0]))
    result=module.with_pair(pred,np.array([.1,.9]),np.array([.2,.8]))
    np.testing.assert_array_equal(result.pred,[2,0])
    np.testing.assert_allclose(result[["p0","p1","p2"]].sum(axis=1),1)
