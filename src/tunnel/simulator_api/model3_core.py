"""모델 ③ 대표 모델 공통 코드 (10_등급2단계_로지스틱.ipynb · 07 노트북과 같은 식).
사진 → 숫자 10개 → (해 높이 + 구름) → 직선 회귀(Ridge α10)로 바깥 밝기 log10 L20 추정
→ 추정 밝기 포함 13개로 2단계 로지스틱 (1단계 하 vs 위험, 2단계 중 vs 상) → 입구 순응 위험도 하·중·상.
입구(암순응)만 다룬다. 출구는 대표 모델 범위 밖."""
import numpy as np, pandas as pd, cv2

FEATURES = ['sky_mean', 'blue_sky', 'white_ratio', 'shadow_contrast', 'road_mean',
            'sky_road_ratio', 'portal_ratio', 'saturation', 'spread', 'mean_all']
PLUS = FEATURES + ['sun_alt', 'cloud']          # 밝기 추정 입력 12개
MAIN_COLS = PLUS + ['est_L']                    # 등급 입력 13개 (추정 밝기 포함)
LAB = ['하', '중', '상']


def photo_features_from_bgr(img, size=640):
    """사진(BGR 배열) 한 장 → 숫자 10개. 07 노트북 photo_features와 같은 식 (터널 표시 없음 → 가운데 어두운 15%)."""
    h, w = img.shape[:2]; s = size / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    H, S, V = [hsv[..., i].astype(float) for i in range(3)]
    sky, road = gray[: h // 3], gray[2 * h // 3:]
    blue = ((H[: h // 3] >= 90) & (H[: h // 3] <= 130) & (S[: h // 3] >= 50) & (V[: h // 3] >= 100)).mean()
    c = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    thr = np.percentile(c, 15)
    mask = np.zeros_like(gray, bool)
    mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = c <= thr
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool) & ~mask
    portal = gray[mask].mean() + 1 if mask.any() else np.nan
    around = gray[ring].mean() + 1 if ring.any() else np.nan
    return {
        'sky_mean': sky.mean() / 255,
        'blue_sky': blue,
        'white_ratio': (gray >= 250).mean(),
        'shadow_contrast': (np.percentile(road, 90) - np.percentile(road, 10)) / 255,
        'road_mean': road.mean() / 255,
        'sky_road_ratio': (sky.mean() + 1) / (road.mean() + 1),
        'portal_ratio': np.log10(around / portal) if portal == portal else np.nan,
        'saturation': S.mean() / 255,
        'spread': (np.percentile(gray, 95) - np.percentile(gray, 5)) / 255,
        'mean_all': gray.mean() / 255,
    }


def photo_features_from_bytes(b):
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('사진을 읽을 수 없음')
    return photo_features_from_bgr(img)


def L20_from_E(E):                      # 일사량(MJ/m²·h) → 바깥 밝기(cd/m²)
    return E * 277.8 * 126 * 0.15 / np.pi


def ratio_from_L20(L20, cls='입구'):    # 바깥 밝기 → 휘도비 (입구 = 지침 경계부 280 × 조광 단계)
    L20 = np.asarray(L20, float)
    step = np.clip(np.ceil(L20 / 4000 * 4) / 4, 0.25, 1)
    return np.where(np.asarray(cls) == '출구', L20 / 45, L20 / (280 * step))


def grade(R):                           # 10배 미만 하 / 10~15 중 / 15 초과 상
    return np.select([np.asarray(R) < 10, np.asarray(R) <= 15], ['하', '중'], '상')


def sun_altitude(ts_kst, lat, lon):     # 해 높이(도), NOAA 근사식. ts_kst = 한국 시각
    t = pd.to_datetime(pd.Series(ts_kst)) - pd.Timedelta(hours=9)
    doy = t.dt.dayofyear.values; hr = (t.dt.hour + t.dt.minute / 60).values
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    decl = (0.006918 - 0.399912*np.cos(g) + 0.070257*np.sin(g) - 0.006758*np.cos(2*g)
            + 0.000907*np.sin(2*g) - 0.002697*np.cos(3*g) + 0.00148*np.sin(3*g))
    eqt = 229.18 * (0.000075 + 0.001868*np.cos(g) - 0.032077*np.sin(g) - 0.014615*np.cos(2*g) - 0.040849*np.sin(2*g))
    tst = hr * 60 + eqt + 4 * np.asarray(lon)
    ha = np.radians(tst / 4 - 180); la = np.radians(np.asarray(lat))
    cosz = np.sin(la)*np.sin(decl) + np.cos(la)*np.cos(decl)*np.cos(ha)
    return np.degrees(np.arcsin(np.clip(cosz, -1, 1)))
