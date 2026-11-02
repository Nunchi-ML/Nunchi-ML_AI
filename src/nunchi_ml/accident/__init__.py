"""모델 ① 고속도로 구간별 사고 위험도 예측 패키지."""
from nunchi_ml.accident.predict import AccidentRiskPredictor, predict_risk

__all__ = ["AccidentRiskPredictor", "predict_risk"]
