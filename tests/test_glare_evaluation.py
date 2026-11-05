import numpy as np
import pytest

from nunchi_ml.glare.evaluation import binary_scores,evaluate_grades


def test_hard_labels_never_become_fake_probabilities():
    result=evaluate_grades([0,1,2],[0,1,1])
    assert result["macro_ovr_auc"] is None
    assert result["probability_status"]=="not_saved"
    assert result["support_2"]==1


def test_good_ranking_and_bad_final_decisions_are_distinct():
    p=np.array([[.8,.15,.05],[.6,.3,.1],[.5,.3,.2]])
    result=evaluate_grades([0,1,2],[0,0,0],p)
    assert result["alert_roc_auc"]==1
    assert result["macro_f1"]==pytest.approx(1/6)
    assert result["multiclass_brier"]==pytest.approx(np.mean(np.sum((p-np.eye(3))**2,axis=1)))


def test_conditional_B_is_evaluated_only_on_original_positive_grades():
    a=np.array([.8,.6,.7,.5]);b=np.array([.99,.1,.9,.2])
    p=np.column_stack([1-a,a*(1-b),a*b])
    result=evaluate_grades([0,1,2,1],[2,1,2,1],p,b)
    assert result["conditional_severe_roc_auc"]==1
    assert result["conditional_severe_support"]==result["conditional_severe_coverage"]==3
    with pytest.raises(ValueError,match="disagree"):
        evaluate_grades([0,1,2,1],[2,1,2,1],p,np.full(4,.5))


def test_undefined_single_class_auc_is_missing_not_zero():
    result=binary_scores([1,1],[.2,.7])
    assert result["roc_auc"] is result["average_precision"] is None
    assert result["brier"]==pytest.approx((.8**2+.3**2)/2)


@pytest.mark.parametrize("probability",[np.array([[.5,.5,.5]]),np.array([[1.,-.1,.1]]),np.array([[np.nan,0,1]])])
def test_invalid_probabilities_rejected(probability):
    with pytest.raises(ValueError,match="normalized"):
        evaluate_grades([0],[0],probability)


def test_zero_alert_mass_does_not_invent_conditional_B():
    result=evaluate_grades([1,2],[0,2],[[1,0,0],[0,0,1]])
    assert result["conditional_severe_coverage"]==1
    assert result["conditional_severe_roc_auc"] is None


def test_float32_roundoff_is_recorded_and_large_errors_rejected():
    result=evaluate_grades([0],[0],[[.5,.3,.19999994]])
    assert 0 < result["probability_row_sum_error_max"] < 1e-6
    assert 0 < result["probability_roundoff_max"] < 1e-6
    with pytest.raises(ValueError,match="normalized"):
        evaluate_grades([0],[0],[[.5,.3,.199]])
