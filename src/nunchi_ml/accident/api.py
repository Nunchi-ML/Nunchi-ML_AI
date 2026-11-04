"""FastAPI application for Model 1 (Highway Glare Accident Risk Serving).

Endpoints:
    - GET  /health              : Server status & model metadata
    - POST /predict             : Single segment risk prediction
    - POST /predict/batch       : Batch segments risk prediction
    - GET  /segments/{route}/{km}: Lookup precalculated/real-time risk by route & km
"""
from contextlib import asynccontextmanager
from typing import List, Optional, Literal
from pathlib import Path
import time
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

from nunchi_ml.accident.predict import AccidentRiskPredictor, BASE_REQUIRED_FEATURES

# 전역 모델 싱글톤 로딩
predictor: Optional[AccidentRiskPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = AccidentRiskPredictor()
    yield


app = FastAPI(
    title="SunSight Model 1 Accident Risk API",
    description="고속도로 1km 구간별 태양 역광 사고 위험도 예측 및 서빙 API (nunchi_ml.accident)",
    version="1.0.0",
    lifespan=lifespan,
)


class SegmentInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    segment_id: Optional[str] = Field(None, description="구간 고유 식별자 (예: GYEONGBU_10KM)")
    route: Optional[str] = Field(None, description="노선명 (예: 경부선)")
    segment_km: Optional[int] = Field(None, description="이정표 km 지점 (예: 10)")
    segment_length_km: float = Field(..., ge=0.01, le=5.0, description="구간 길이 (km)")
    is_partial_segment: int = Field(0, ge=0, le=1, description="불완전 구간 여부 (0 또는 1)")
    curvature_mean: float = Field(..., ge=0.0, description="평균 곡률")
    curvature_max: float = Field(..., ge=0.0, description="최대 곡률")
    min_tunnel_dist_km: float = Field(..., ge=0.0, description="가장 가까운 터널과의 거리 (km)")
    has_tunnel_within_1km: int = Field(..., ge=0, le=1, description="1km 이내 터널 존재 여부 (0 또는 1)")
    tunnel_portals_1km: int = Field(..., ge=0, description="1km 이내 터널 포털(입·출구) 개수")
    max_tunnel_length_1km: float = Field(..., ge=0.0, description="1km 이내 가장 긴 터널의 길이 (m)")
    ew_alignment: float = Field(..., ge=0.0, le=1.0, description="동서축 정렬도 (|sin(bearing)|)")
    annual_glare_hours_fwd: float = Field(..., ge=0.0, description="정방향 연간 잠재 역광 시간 (h)")
    commute_glare_hours: float = Field(..., ge=0.0, description="출퇴근 시간대 잠재 역광 시간 (h)")


class PredictionOutput(BaseModel):
    segment_id: Optional[str] = None
    predicted_accident_risk: float = Field(..., description="연간 예상 사고 위험 지수 (Poisson Rate, 절대 사고 확률이 아님)")
    risk_grade: Literal["SAFE", "WARNING", "DANGER"] = Field(..., description="위험도 등급")
    description: str = Field(..., description="위험 등급 한글 설명")
    architecture: str = Field(..., description="추론 모델 구조 (예: lightgbm_single)")
    model_version: str = Field(..., description="모델 패키지 버전")
    inference_time_ms: float = Field(..., description="추론 소요 시간 (밀리초)")


class BatchPredictionInput(BaseModel):
    segments: List[SegmentInput] = Field(..., max_length=10000, description="예측할 도로 구간 목록")


class BatchPredictionOutput(BaseModel):
    total_count: int
    predictions: List[PredictionOutput]
    total_latency_ms: float


@app.get("/health", tags=["System"])
def health_check():
    """서버 헬스체크 및 로드된 모델 메타데이터 반환."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="모델이 아직 메모리에 로드되지 않았습니다."
        )
    return {
        "status": "healthy",
        "service": "SunSight Model 1 (Accident Risk)",
        "model_name": predictor.model_name,
        "model_version": predictor.version,
        "architecture": predictor.architecture,
        "thresholds": {
            "warning": predictor.threshold_warning,
            "danger": predictor.threshold_danger,
        },
        "required_features": BASE_REQUIRED_FEATURES,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_single(data: SegmentInput):
    """단일 1km 도로 구간 사고 위험도 및 등급 예측."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    t0 = time.perf_counter()
    try:
        res = predictor.predict_segment(data.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"추론 실패: {e}")
    latency = (time.perf_counter() - t0) * 1000

    res["segment_id"] = data.segment_id
    res["inference_time_ms"] = round(latency, 3)
    return res


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Inference"])
def predict_batch_api(data: BatchPredictionInput):
    """다수 도로 구간 일괄 고속 예측."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    t0 = time.perf_counter()
    raw_dicts = [s.model_dump() for s in data.segments]
    df = pd.DataFrame(raw_dicts)

    try:
        pred_df = predictor.predict_batch(df)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"배치 추론 실패: {e}")

    results = []
    for i, row in pred_df.iterrows():
        grade = str(row["risk_grade"])
        if grade == "DANGER":
            desc = "고위험 구간 (태양 직광 + 터널 입출구/곡선부 집중)"
        elif grade == "WARNING":
            desc = "주의 구간 (특정 계절/시간대 눈부심 노출)"
        else:
            desc = "안전 구간 (상대적 사고 위험 낮음)"

        results.append(
            PredictionOutput(
                segment_id=row.get("segment_id"),
                predicted_accident_risk=float(row["pred_accident_risk"]),
                risk_grade=grade,
                description=desc,
                architecture=predictor.architecture,
                model_version=predictor.version,
                inference_time_ms=0.0,
            )
        )

    total_latency = (time.perf_counter() - t0) * 1000
    return BatchPredictionOutput(
        total_count=len(results),
        predictions=results,
        total_latency_ms=round(total_latency, 2),
    )
