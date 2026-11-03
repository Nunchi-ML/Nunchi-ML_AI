"""눈치 드라이브 ↔ 머신러닝 (모델 ① 사고위험 + 모델 ② 눈부심감지) 실시간 연동 시뮬레이터 데모.

실행:
    python -m src.demo_live_drive
"""
import os
import sys
import time
import glob
import cv2
import numpy as np
import pandas as pd
import joblib

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
from nunchi_ml.accident.predict import predict_risk
from nunchi_ml.glare.features import extract

MODEL2_PATH = os.path.join(BASE_DIR, "models", "model2-pilot.joblib")


def predict_glare_image(img_path, model2_pkg):
    """전방 카메라 이미지 한 장으로부터 눈부심 등급(0/1/2) 및 확률 예측."""
    feats_dict = extract(img_path)
    df_f = pd.DataFrame([feats_dict])[model2_pkg["features"]]
    pred_cls = int(model2_pkg["model"].predict(df_f)[0])
    probs = model2_pkg["model"].predict_proba(df_f)[0]
    return pred_cls, probs[pred_cls], feats_dict


def main():
    print("=" * 75)
    print("🚘 [Nunchi Drive] 실시간 머신러닝 연동 주행 시뮬레이션 데모 시작")
    print("   * 모델 ①: 도로 환경 & 태양 위치 기반 실시간 사고 위험도 예측 (LightGBM)")
    print("   * 모델 ②: 전방 카메라 영상 OpenCV 12개 피처 실시간 눈부심 감지 (OpenCV + ML)")
    print("=" * 75)
    time.sleep(1)

    model2_pkg = joblib.load(MODEL2_PATH)

    # 실제 GLARE 데이터셋에서 정상 / 심한 눈부심 샘플 이미지 자동 탐색
    all_imgs = sorted(glob.glob(os.path.join(BASE_DIR, "data", "glare", "Images", "*", "*", "*.png")))
    if not all_imgs:
        raise FileNotFoundError("GLARE 이미지를 찾을 수 없습니다.")

    # 대표 샘플 선정 (정상 프레임과 직광 프레임)
    sample_normal = all_imgs[0]
    sample_glare = all_imgs[min(50, len(all_imgs)-1)]

    # 가상 주행 시나리오 (대구 달구벌대로 및 터널 구간 5단계)
    scenarios = [
        {
            "step": 1,
            "dist": "0.0 km",
            "desc": "도심 평지 구간 주행 (오전 08:00, 동향 주행)",
            "road_info": {
                "segment_length_km": 1.0,
                "curvature_mean": 0.001,
                "has_tunnel_within_1km": 0,
                "ew_alignment": 0.20,
                "commute_glare_hours": 5.0,
            },
            "img_path": sample_normal,
        },
        {
            "step": 2,
            "dist": "1.5 km",
            "desc": "터널 진입로 접근 중 (800m 전방 터널)",
            "road_info": {
                "segment_length_km": 1.0,
                "curvature_mean": 0.003,
                "has_tunnel_within_1km": 1,
                "min_tunnel_dist_km": 0.3,
                "ew_alignment": 0.70,
                "commute_glare_hours": 18.0,
            },
            "img_path": sample_normal,
        },
        {
            "step": 3,
            "dist": "2.8 km",
            "desc": "🚨 터널 출구 통과 직후 (동쪽 저각도 태양 정면 직광 노출!)",
            "road_info": {
                "segment_length_km": 1.0,
                "curvature_mean": 0.006,
                "curvature_max": 0.012,
                "has_tunnel_within_1km": 1,
                "min_tunnel_dist_km": 0.05,
                "ew_alignment": 0.98,
                "annual_glare_hours_fwd": 60.0,
                "commute_glare_hours": 32.0,
            },
            "img_path": sample_glare,
        },
        {
            "step": 4,
            "dist": "4.0 km",
            "desc": "직광 통과 후 곡선 도로 진입",
            "road_info": {
                "segment_length_km": 1.0,
                "curvature_mean": 0.004,
                "has_tunnel_within_1km": 0,
                "ew_alignment": 0.40,
                "commute_glare_hours": 12.0,
            },
            "img_path": sample_normal,
        },
    ]

    for sc in scenarios:
        print(f"\n📍 [주행 위치 {sc['dist']}] : {sc['desc']}")
        time.sleep(0.8)

        # 1. 모델 ① 사고 위험도 추론
        t0 = time.time()
        risk_res = predict_risk(sc["road_info"])
        t1 = time.time()

        grade = risk_res["risk_grade"]
        score = risk_res["predicted_accident_risk"]
        latency_m1 = (t1 - t0) * 1000

        badge_m1 = {
            "SAFE": "🟢 [SAFE 안전]",
            "WARNING": "🟡 [WARNING 주의]",
            "DANGER": "🔴 [DANGER 고위험]",
        }.get(grade, grade)

        print(f"  ├─ [모델 ① 도로 위험도] {badge_m1} 위험점수: {score:.4f} (추론: {latency_m1:.1f}ms)")
        if grade == "DANGER":
            print(f"  │  📢 HUD 내비게이션: \"위험 구간입니다! 터널 출구 급격한 눈부심에 주의하여 서행하세요!\"")
        elif grade == "WARNING":
            print(f"  │  📢 HUD 내비게이션: \"전방 1km 이내 터널 진출입 주의 구간입니다.\"")

        # 2. 모델 ② 전방 카메라 눈부심 판정
        t2 = time.time()
        g_cls, g_conf, g_feats = predict_glare_image(sc["img_path"], model2_pkg)
        t3 = time.time()
        latency_m2 = (t3 - t2) * 1000

        glare_labels = {
            0: "정상 시야 (Normal)",
            1: "눈부심 발생 (Glare)",
            2: "심한 눈부심 (Severe Glare - 시야 차단)",
        }

        print(f"  └─ [모델 ② 카메라 비전] 결과: Class {g_cls} ({glare_labels[g_cls]}) | 신뢰도: {g_conf*100:.1f}% (추론: {latency_m2:.1f}ms)")
        print(f"     * 주요 광학 피처: 과노출 픽셀 {g_feats['white_ratio']*100:.1f}%, 광원 밝기 {g_feats['brightest_val']:.0f}/255, 도로 선명도 {g_feats['road_sharp']:.1f}")

        # 3. 게임(FE) 스마트 선팅 제어 반응 시뮬레이션
        if g_cls == 2 or grade == "DANGER":
            tint_pct = 80
            print(f"     ⚡ [차량 제어 디바이스] 스마트 선팅 유리 작동: 전면 유리 투과율 100% ➔ {100-tint_pct}% (차광률 {tint_pct}% 가동!)")
        elif g_cls == 1:
            tint_pct = 40
            print(f"     ⚡ [차량 제어 디바이스] 스마트 선팅 유리 작동: 전면 유리 차광률 {tint_pct}% 가동")
        else:
            print(f"     ⚡ [차량 제어 디바이스] 스마트 선팅 유리 대기: 투명 상태 유지 (0%)")

        time.sleep(1.2)

    print("\n" + "=" * 75)
    print("🎉 [시뮬레이션 완료] 모델 ①(사고 예측)과 모델 ②(비전 차광)가 실시간으로 성공적으로 연동되었습니다.")
    print("=" * 75)


if __name__ == "__main__":
    main()
