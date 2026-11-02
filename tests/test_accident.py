"""모델 ① 고속도로 구간 사고 위험도 예측 및 입력 검증 테스트.

백엔드 연동 시 입력값 결측, 자료형 오류, 무한대/NaN 등의 오류 차단 및
실제 학습 모델 아티팩트(models/model1-v1.joblib) 로딩과 예측 정합성을 검증한다.
"""
from pathlib import Path
import pytest
import numpy as np
from nunchi_ml.accident.predict import AccidentRiskPredictor, predict_risk, DEFAULT_MODEL_PATH


VALID_SEGMENT = {
    "segment_length_km": 1.0,
    "is_partial_segment": 0,
    "curvature_mean": 0.0012,
    "curvature_max": 0.0035,
    "min_tunnel_dist_km": 0.45,
    "has_tunnel_within_1km": 1,
    "tunnel_portals_1km": 2,
    "max_tunnel_length_1km": 520.0,
    "ew_alignment": 0.88,
    "annual_glare_hours_fwd": 42.5,
    "commute_glare_hours": 18.0,
}


def test_predict_risk_valid_input():
    """정상 입력 시 사고 위험 지수 및 유효 등급(SAFE/WARNING/DANGER) 반환 검증."""
    result = predict_risk(VALID_SEGMENT)
    assert "predicted_accident_risk" in result
    assert "risk_grade" in result
    assert result["risk_grade"] in ["SAFE", "WARNING", "DANGER"]
    assert isinstance(result["predicted_accident_risk"], float)
    assert result["predicted_accident_risk"] > 0


def test_predict_risk_missing_feature_raises_error():
    """성진 님 요청: 필수 피처가 하나라도 누락되면 0 채움 없이 ValueError 발생."""
    invalid_segment = VALID_SEGMENT.copy()
    del invalid_segment["ew_alignment"]

    with pytest.raises(ValueError, match="누락"):
        predict_risk(invalid_segment)


@pytest.mark.parametrize("invalid_val", [float("nan"), float("inf"), float("-inf"), None])
def test_predict_risk_nan_or_inf_raises_error(invalid_val):
    """NaN, None, inf, -inf 등 비정상 수치 입력 시 차단 검증."""
    invalid_segment = VALID_SEGMENT.copy()
    invalid_segment["commute_glare_hours"] = invalid_val

    with pytest.raises(ValueError):
        predict_risk(invalid_segment)


def test_predict_risk_non_numeric_type_raises_error():
    """문자열 등 비수치형 데이터 입력 시 차단 검증."""
    invalid_segment = VALID_SEGMENT.copy()
    invalid_segment["curvature_mean"] = "high"

    with pytest.raises(ValueError, match="숫자형"):
        predict_risk(invalid_segment)


def test_model_artifact_loading_and_prediction():
    """실제 joblib 모델 파일 로딩 및 내부 메타데이터, 예측 정합성 검증."""
    assert Path(DEFAULT_MODEL_PATH).exists(), f"모델 파일이 존재하지 않습니다: {DEFAULT_MODEL_PATH}"

    predictor = AccidentRiskPredictor(DEFAULT_MODEL_PATH)
    assert hasattr(predictor, "lgb_model")
    assert predictor.lgb_model is not None
    assert len(predictor.feature_names) == 12  # 11개 기본 피처 + 1개 교호작용 피처
    assert "tunnel_glare_interaction" in predictor.feature_names

    res = predictor.predict_segment(VALID_SEGMENT)
    assert res["model_version"] == "1.0.0"
    assert res["architecture"] == "lightgbm_single"
    assert res["predicted_accident_risk"] > 0
