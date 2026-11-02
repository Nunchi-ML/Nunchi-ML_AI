"""고속도로 1km 구간 사고 위험도(모델 ①) 백엔드 연동용 서빙/추론 모듈.

사용 예시:
    from src.accident.predict import AccidentRiskPredictor

    predictor = AccidentRiskPredictor()
    result = predictor.predict_segment({
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
    })
    print(result)
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "model1-v1.joblib")


class AccidentRiskPredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}. 먼저 train_final.py를 실행하세요.")
        
        self.package = joblib.load(model_path)
        self.model_name = self.package.get("model_name", "model1")
        self.version = self.package.get("version", "1.0.0")
        self.feature_names = self.package["feature_names"]
        self.architecture = self.package.get("chosen_architecture", "lightgbm_single")
        self.lgb_model = self.package["lgb_model"]
        self.glm_model = self.package.get("glm_model")
        self.scaler = self.package.get("scaler")
        self.weights = self.package.get("ensemble_weights", {"lgb": 1.0, "glm": 0.0})

        # 위험 등급 기준값 (5,782개 구간 전체 예측 분포 기반)
        # 상위 25% (0.025 이상): 주의(WARNING), 상위 10% (0.035 이상): 고위험(DANGER)
        self.threshold_warning = 0.025
        self.threshold_danger = 0.035

    def _prepare_features(self, data):
        """딕셔너리 또는 DataFrame으로부터 모델 피처 배열 생성 및 교호작용 계산."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("입력 데이터는 dict 또는 pandas.DataFrame이어야 합니다.")

        # 파생 변수(터널 x 출퇴근 역광 교호작용) 자동 계산
        if "tunnel_glare_interaction" not in df.columns:
            has_tun = df["has_tunnel_within_1km"] if "has_tunnel_within_1km" in df.columns else 0
            commute = df["commute_glare_hours"] if "commute_glare_hours" in df.columns else 0
            df["tunnel_glare_interaction"] = has_tun * commute

        # 누락된 기본 피처 기본값 채우기
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0

        return df[self.feature_names]

    def predict_segment(self, segment_dict: dict) -> dict:
        """단일 1km 도로 구간의 연간 예상 눈부심 사고 위험 지수 및 위험도 등급 반환."""
        df_feat = self._prepare_features(segment_dict)
        lgb_pred = np.clip(self.lgb_model.predict(df_feat)[0], 1e-6, None)

        if self.architecture == "ensemble" and self.glm_model and self.scaler:
            x_scaled = self.scaler.transform(df_feat)
            glm_pred = np.clip(self.glm_model.predict(x_scaled)[0], 1e-6, None)
            w_lgb = self.weights.get("lgb", 0.7)
            w_glm = self.weights.get("glm", 0.3)
            raw_risk = float(w_lgb * lgb_pred + w_glm * glm_pred)
        else:
            raw_risk = float(lgb_pred)

        # 위험도 등급 산정
        if raw_risk >= self.threshold_danger:
            risk_grade = "DANGER"
            grade_desc = "고위험 구간 (태양 직광 + 터널 입출구/곡선부 집중)"
        elif raw_risk >= self.threshold_warning:
            risk_grade = "WARNING"
            grade_desc = "주의 구간 (특정 계절/시간대 눈부심 노출)"
        else:
            risk_grade = "SAFE"
            grade_desc = "안전 구간 (상대적 사고 위험 낮음)"

        return {
            "predicted_accident_risk": round(raw_risk, 5),
            "risk_grade": risk_grade,
            "description": grade_desc,
            "architecture": self.architecture,
            "model_version": self.version,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """대량 구간 DataFrame에 대해 일괄 예측 컬럼 추가."""
        df_feat = self._prepare_features(df)
        lgb_preds = np.clip(self.lgb_model.predict(df_feat), 1e-6, None)

        if self.architecture == "ensemble" and self.glm_model and self.scaler:
            x_scaled = self.scaler.transform(df_feat)
            glm_preds = np.clip(self.glm_model.predict(x_scaled), 1e-6, None)
            w_lgb = self.weights.get("lgb", 0.7)
            w_glm = self.weights.get("glm", 0.3)
            final_preds = w_lgb * lgb_preds + w_glm * glm_preds
        else:
            final_preds = lgb_preds

        result_df = df.copy()
        result_df["pred_accident_risk"] = np.round(final_preds, 5)
        result_df["risk_grade"] = np.where(
            final_preds >= self.threshold_danger, "DANGER",
            np.where(final_preds >= self.threshold_warning, "WARNING", "SAFE")
        )
        return result_df


def predict_risk(segment_dict: dict) -> dict:
    """간편 호출용 전역 함수."""
    predictor = AccidentRiskPredictor()
    return predictor.predict_segment(segment_dict)
