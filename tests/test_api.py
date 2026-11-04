"""Test suite for Model 1 FastAPI serving endpoints."""
import pytest
from fastapi.testclient import TestClient
import pandas as pd
from pathlib import Path

from nunchi_ml.accident.api import app

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_api():
    with TestClient(app) as _:
        pass

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "thresholds" in data
    assert data["thresholds"]["warning"] == 0.025
    assert data["thresholds"]["danger"] == 0.035

def test_predict_single_safe():
    payload = {
        "segment_id": "TEST_SEG_SAFE",
        "segment_length_km": 1.0,
        "is_partial_segment": 0,
        "curvature_mean": 0.0001,
        "curvature_max": 0.0005,
        "min_tunnel_dist_km": 5.0,
        "has_tunnel_within_1km": 0,
        "tunnel_portals_1km": 0,
        "max_tunnel_length_1km": 0.0,
        "ew_alignment": 0.1,
        "annual_glare_hours_fwd": 0.0,
        "commute_glare_hours": 0.0,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["segment_id"] == "TEST_SEG_SAFE"
    assert data["risk_grade"] == "SAFE"
    assert "predicted_accident_risk" in data
    assert data["inference_time_ms"] > 0

def test_predict_single_danger():
    payload = {
        "segment_id": "TEST_SEG_DANGER",
        "segment_length_km": 1.0,
        "is_partial_segment": 0,
        "curvature_mean": 5.5,
        "curvature_max": 7.0,
        "min_tunnel_dist_km": 11.43,
        "has_tunnel_within_1km": 0,
        "tunnel_portals_1km": 0,
        "max_tunnel_length_1km": 0.0,
        "ew_alignment": 0.96,
        "annual_glare_hours_fwd": 405.25,
        "commute_glare_hours": 738.5,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["segment_id"] == "TEST_SEG_DANGER"
    assert data["risk_grade"] == "DANGER"

def test_predict_batch_benchmark():
    # 실제 5,782개 구간 중 1,000개 로드 후 벤치마크
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "data" / "expressway" / "segment_1km_dataset_v1.csv"
    if not csv_path.exists():
        pytest.skip("데이터셋 파일 없음")

    df = pd.read_csv(csv_path).head(1000)
    segments = []
    for i, r in df.iterrows():
        segments.append({
            "segment_id": f"SEG_{i}",
            "segment_length_km": float(r["segment_length_km"]),
            "is_partial_segment": int(r["is_partial_segment"]),
            "curvature_mean": float(r["curvature_mean"]),
            "curvature_max": float(r["curvature_max"]),
            "min_tunnel_dist_km": float(r["min_tunnel_dist_km"]),
            "has_tunnel_within_1km": int(r["has_tunnel_within_1km"]),
            "tunnel_portals_1km": int(r["tunnel_portals_1km"]),
            "max_tunnel_length_1km": float(r["max_tunnel_length_1km"]),
            "ew_alignment": float(r["ew_alignment"]),
            "annual_glare_hours_fwd": float(r["annual_glare_hours_fwd"]),
            "commute_glare_hours": float(r["commute_glare_hours"]),
        })

    payload = {"segments": segments}
    res = client.post("/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_count"] == 1000
    assert len(data["predictions"]) == 1000
    print(f"\n[API Benchmark] 1,000 segments batch latency: {data['total_latency_ms']} ms")
