import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'notebooks/experiments/sun_visible_probe'))
    return importlib.import_module('sd_photo_validation')


def test_vectorized_rule_matches_exhaustive_training_grid_and_tie_order(module):
    frame = pd.DataFrame(dict(x=[0,1,2,3,4,5,6,7,8], z=[8,2,7,1,6,0,5,3,4],label=[2,2,1,1,0,0,0,0,0]))
    expected = None
    for col in ['z','x']:
        qs = np.unique(np.quantile(frame[col],np.linspace(.05,.95,19)))
        for i,a in enumerate(qs):
            for b in qs[i+1:]:
                for sign in [1,-1]:
                    pred = np.digitize(frame[col],[a,b]) if sign == 1 else np.digitize(-frame[col],[-b,-a])
                    score = f1_score(frame.label,pred,labels=[0,1,2],average='macro',zero_division=0)
                    if expected is None or score > expected[0]:
                        expected = score,col,a,b,sign,pred
    actual = module.fit_rule(frame,['z','x'])
    assert actual['feature'] == expected[1] and actual['sign'] == expected[4]
    np.testing.assert_array_equal(module.apply_rule(frame,actual),expected[5])
    assert actual['training_macro_f1'] == pytest.approx(expected[0])


def test_rule_prediction_ignores_test_labels_and_rejects_nonfinite_features(module):
    train = pd.DataFrame(dict(x=range(9),label=[0]*3+[1]*3+[2]*3))
    rule = module.fit_rule(train,['x'])
    a = module.apply_rule(train,rule)
    np.testing.assert_array_equal(a,module.apply_rule(train.assign(label=2),rule))
    with pytest.raises(ValueError,match='Finite'):
        module.apply_rule(pd.DataFrame(dict(x=[np.nan])),rule)
    with pytest.raises(ValueError,match='nonconstant'):
        module.fit_rule(train.assign(x=1),['x'])


def test_abstentions_remain_missing_and_reference_coverage_is_exact(module):
    meta = pd.DataFrame(dict(stable_id=['a','b','c']))
    labels = pd.DataFrame(dict(stable_id=['c','a','b'],status=['complete']*3,grade=['2','0','abstain']))
    result = module.join_reference(meta,labels)
    assert result.stable_id.tolist() == ['a','b','c']
    assert result.accepted_reference.tolist() == [True,False,True]
    assert pd.isna(result.loc[1,'label'])
    assert result.loc[2,'label'] == 2
    for bad in [labels.iloc[:2],pd.concat([labels,labels.iloc[:1]]),labels.assign(status='pending')]:
        with pytest.raises(ValueError):
            module.join_reference(meta,bad)


def test_reference_cannot_silently_accept_fractional_or_unknown_grades(module):
    meta = pd.DataFrame(dict(stable_id=['a']))
    for grade in ['','1.5','unknown','nan']:
        with pytest.raises(ValueError):
            module.join_reference(meta,pd.DataFrame(dict(stable_id=['a'],status=['complete'],grade=[grade])))


def test_reproduction_checks_probability_drift_even_with_same_decision(module):
    frame = pd.DataFrame(dict(id=['a'],method=['x'],pred=[0],p0=[.8],p1=[.1],p2=[.1]))
    module.verify_reproduction(frame,frame)
    with pytest.raises(ValueError,match='probabilities'):
        module.verify_reproduction(frame.assign(p0=.7,p1=.2),frame)
    with pytest.raises(ValueError,match='predictions'):
        module.verify_reproduction(frame.assign(pred=1),frame)


def test_intervals_keep_subsets_separate_and_resample_dates_together(module):
    frame = pd.DataFrame(dict(subset=['a','a','b','b'],group_id=['date1']*4,label=[0,2,0,2],
                              rule_all_features=[0,0,0,2],ml=[0,2,0,0]))
    a = module.paired_intervals(frame,['rule_all_features','ml'],repeats=50)
    b = module.paired_intervals(frame,['rule_all_features','ml'],repeats=50)
    assert a == b
    values = {r['subset']:r for r in a if r['method']=='ml'}
    assert values['a']['low'] > 0 and values['b']['high'] < 0
    assert all(r['low']==r['high'] and r['filename_date_groups']==1 for r in a)
