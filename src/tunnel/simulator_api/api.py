"""모델 ③ 대표 모델 API (시뮬레이터용). 터널 입구 순응 위험도 하·중·상.
실행: uvicorn api:app --host 0.0.0.0 --port 8000   (같은 폴더에 model3_대표모델.joblib 필요 → train_model.py로 만듦)
문서: http://localhost:8000/docs"""
from pathlib import Path
from typing import List, Optional
import joblib, numpy as np, pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from model3_core import FEATURES, photo_features_from_bytes, sun_altitude
from pupil import pupil_entrance

M = joblib.load(Path(__file__).with_name('model3_대표모델.joblib'))
WARN_DISTANCE_M = 155      # 국토부 터널 조명 지침 정지시거 (입구 앞 구간)
BOUNDARY_M = 80            # 경계부 길이 (100km/h에서 약 2.9초)
MSG = {'하': '터널 입구 밝기 차이가 크지 않아요.',
       '중': '터널 입구에서 눈이 어둠에 적응하기 조금 힘들어요. 속도를 줄여 주세요.',
       '상': '터널 입구에서 눈이 어둠에 적응하기 매우 힘들어요. 미리 속도를 줄여 주세요.'}
EN = {'하': 'low', '중': 'mid', '상': 'high'}
# 화면 어둡기 (11/7 결정): 시작 어둡기 = 등급, 회복 = 3초 고정
DARK_START = {'하': 0.30, '중': 0.60, '상': 0.85}
RECOVER_S = 3.0


def screen_effect(g):
    """터널 입구 통과 순간 화면을 덮는 검은 막의 진하기(0~1)와 3초 회복 곡선. 0.5초 간격, 부드럽게(ease-out) 줄어듦."""
    d0 = DARK_START[g]; ts = np.arange(0, RECOVER_S + 1e-9, 0.5)
    curve = [{'t_s': float(t), 'dark_opacity': round(float(d0 * (1 - t / RECOVER_S) ** 2), 3)} for t in ts]
    return {'start_dark_opacity': d0, 'recover_s': RECOVER_S, 'countdown': [3, 2, 1], 'curve': curve}

app = FastAPI(title='SunSight 모델 ③ 터널 입구 순응 위험도 API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


def _predict(rows: pd.DataFrame, speed_kmh=100) -> List[dict]:
    d = rows.copy()
    for c in M['PLUS']:
        if c not in d: d[c] = np.nan            # 빈칸 → 학습 중앙값 (모델 안에서 채움)
    d['est_L'] = M['ridge'].predict(d[M['PLUS']])
    X = d[M['MAIN_COLS']].to_numpy(float)
    p1 = M['stage1'].predict_proba(X)[:, 1]; p2 = M['stage2'].predict_proba(X)[:, 1]
    r1 = M['stage1'].predict(X).astype(bool); r2 = M['stage2'].predict(X).astype(bool)
    out = []
    for i in range(len(d)):
        g = '하' if not r1[i] else ('상' if r2[i] else '중')
        L20 = float(10 ** d.est_L.iloc[i])
        step = float(np.clip(np.ceil(L20 / 4000 * 4) / 4, 0.25, 1))
        inside = 280 * step
        out.append({
            'grade': g, 'grade_en': EN[g],
            'grade_prob': {'하': round(float(1 - p1[i]), 3), '중': round(float(p1[i] * (1 - p2[i])), 3), '상': round(float(p1[i] * p2[i]), 3)},
            'est_outside_luminance_cd_m2': round(L20, 1),
            'tunnel_entrance_luminance_cd_m2': round(inside, 1),
            'est_luminance_ratio': round(L20 / inside, 2),
            'sun_altitude_deg': None if pd.isna(d.sun_alt.iloc[i]) else round(float(d.sun_alt.iloc[i]), 1),
            'cloud': None if pd.isna(d.cloud.iloc[i]) else float(d.cloud.iloc[i]),
            'message': MSG[g], 'warn_distance_m': WARN_DISTANCE_M, 'boundary_zone_m': BOUNDARY_M,
            'scope': '터널 입구(암순응)만. 출구는 범위 밖',
            'screen_effect': screen_effect(g),
            'pupil': pupil_entrance(L20, inside, 40, '기본', speed_kmh),   # 같은 추정 밝기로 동공 계산 (설명용, 40세 · τ2 기본 고정)
        })
    return out


def _sun(captured_at, lat, lon, sun_alt):
    if sun_alt is not None: return sun_alt
    if captured_at is None or lat is None or lon is None:
        raise HTTPException(422, 'sun_alt 또는 captured_at+lat+lon 중 하나가 필요함')
    return float(sun_altitude([captured_at], lat, lon)[0])


@app.get('/health')
def health():
    return {'ok': True, 'model': '추정 밝기(Ridge α10) + 2단계 로지스틱 C=0.01', 'n_train': M['n_train'], 'sklearn': M['sklearn']}


@app.post('/predict/photo')
async def predict_photo(image: UploadFile = File(..., description='터널 입구가 보이는 앞 유리 사진'),
                        captured_at: Optional[str] = Form(None, description='한국 시각, 예 2026-11-10T08:20:00'),
                        lat: Optional[float] = Form(None), lon: Optional[float] = Form(None),
                        cloud: Optional[float] = Form(None, description='ASOS 전운량 0~10, 모르면 비움'),
                        sun_alt: Optional[float] = Form(None, description='해 높이(도). 주면 captured_at·lat·lon 안 씀'),
                        speed_kmh: float = Form(100, description='차 속도 (동공 설명 칸의 경계부 시간에만 씀)')):
    _check(speed_kmh)
    try:
        f = photo_features_from_bytes(await image.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    f['sun_alt'] = _sun(captured_at, lat, lon, sun_alt); f['cloud'] = np.nan if cloud is None else cloud
    r = _predict(pd.DataFrame([f]), speed_kmh)[0]; r['photo_features'] = {k: (None if pd.isna(f[k]) else round(float(f[k]), 4)) for k in FEATURES}
    return r


class FeatureRow(BaseModel):
    id: Optional[str] = Field(None, description='시뮬레이터 쪽 터널·장면 이름')
    features: dict = Field(..., description='사진 숫자 10개 (빠진 칸은 학습 중앙값)')
    captured_at: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    sun_alt: Optional[float] = None
    cloud: Optional[float] = None
    speed_kmh: float = 100


def _check(speed_kmh):
    if speed_kmh <= 0: raise HTTPException(422, 'speed_kmh > 0')


@app.post('/predict/features')
def predict_features(rows: List[FeatureRow]):
    recs = []
    for r in rows:
        x = {k: r.features.get(k, np.nan) for k in FEATURES}
        x['sun_alt'] = _sun(r.captured_at, r.lat, r.lon, r.sun_alt); x['cloud'] = np.nan if r.cloud is None else r.cloud
        recs.append(x)
    res = []
    for r, x in zip(rows, recs):
        _check(r.speed_kmh)
        o = _predict(pd.DataFrame([x]), r.speed_kmh)[0]; o['id'] = r.id; res.append(o)
    return res
