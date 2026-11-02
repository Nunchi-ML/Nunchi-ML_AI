"""모델 ② 공통 전처리 — BE가 같은 코드로 예측하므로 깨지면 서버 예측이 틀어진다."""
import hashlib
import importlib.resources

import numpy as np

from nunchi_ml.glare.features import FEATURES, extract_array
from nunchi_ml.glare.inference import features_sha256

# 배포된 모델(model2-pilot-v1)의 meta.json 에 고정된 features.py 지문.
# features.py 를 한 글자라도 바꾸면 기존 모델을 서버에서 못 불러온다 → 바꿀 땐 모델을 다시 export 하고 이 값도 같이 바꿀 것.
RELEASED_FEATURES_SHA256 = "9e0e673dfaed53178f20070c57a8e6a0f3f7a692d718be88825bd49c10c4224e"


def test_features_file_matches_released_model():
    assert features_sha256() == RELEASED_FEATURES_SHA256
    source = importlib.resources.files("nunchi_ml.glare").joinpath("features.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == RELEASED_FEATURES_SHA256


def test_extract_gives_every_feature_as_finite_number():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (368, 640, 3), dtype=np.uint8)
    out = extract_array(image)
    assert set(FEATURES) <= set(out)
    assert all(np.isfinite(float(out[k])) for k in FEATURES)


def test_brighter_frame_has_higher_mean_and_more_white():
    dark = np.full((368, 640, 3), 40, dtype=np.uint8)
    bright = np.full((368, 640, 3), 250, dtype=np.uint8)
    d, b = extract_array(dark), extract_array(bright)
    assert b["mean_all"] > d["mean_all"]
    assert b["white_ratio"] >= d["white_ratio"]
