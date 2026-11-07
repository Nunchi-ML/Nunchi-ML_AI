# SunSight 모델 ③ API (시뮬레이터용)

터널 입구 사진 한 장 + 촬영 시각·위치 (+ 구름)를 넣으면, **햇빛 때문에 눈의 순응 부담이 생기는 터널 입구의 위험도**를 하 · 중 · 상으로 돌려준다.
사고 장소·시각을 예측하는 API가 아니다. 터널 입구(밝은 곳 → 어두운 곳)만 다루고, 출구는 범위 밖이다.

## 1. 실행

```bash
pip install -r requirements.txt
python train_model.py --train ../sunsight_video/data/aihub_71786/학습표_71786.csv   # model3_대표모델.joblib 생성 (한 번만)
uvicorn api:app --host 0.0.0.0 --port 8000
```

- 브라우저에서 `http://localhost:8000/docs`를 열면 입력 칸과 예시가 있는 시험 화면이 나온다.
- scikit-learn 버전이 바뀌면 `train_model.py`를 다시 돌려 joblib을 새로 만든다.

## 2. 엔드포인트

| 주소 | 용도 | 입력 |
| --- | --- | --- |
| `GET /health` | 살아 있는지 확인 | 없음 |
| `POST /predict/photo` | 사진으로 바로 판정 (권장) | multipart: `image`(jpg/png), `captured_at`(한국 시각 `2026-11-10T08:20:00`), `lat`, `lon`, `cloud`(ASOS 전운량 0~10, 모르면 비움). `captured_at·lat·lon` 대신 `sun_alt`(해 높이, 도)만 줘도 된다 |
| `POST /predict/features` | 사진 숫자를 미리 계산해 둔 경우 (여러 장 한 번에) | JSON 배열: `[{"id": "터널A_맑음", "features": {...10개...}, "sun_alt": 32.5, "cloud": 2}]` |

### 예시

```bash
curl -X POST http://localhost:8000/predict/photo \
  -F image=@tunnel.jpg -F captured_at=2026-11-10T08:20:00 -F lat=35.87 -F lon=128.63 -F cloud=2
```

```json
{
  "grade": "중", "grade_en": "mid",
  "grade_prob": {"하": 0.416, "중": 0.536, "상": 0.048},
  "est_outside_luminance_cd_m2": 2031.6,
  "tunnel_entrance_luminance_cd_m2": 196.0,
  "est_luminance_ratio": 10.37,
  "sun_altitude_deg": 47.7, "cloud": null,
  "message": "터널 입구에서 눈이 어둠에 적응하기 조금 힘들어요. 속도를 줄여 주세요.",
  "warn_distance_m": 155, "boundary_zone_m": 80,
  "scope": "터널 입구(암순응)만. 출구는 범위 밖",
  "pupil": {"age": 40, "tau2_assumption": "기본", "boundary_time_s": 2.88,
            "pupil_outside_mm": 3.5, "pupil_inside_full_mm": 4.1, "pupil_boundary_end_mm": 3.66,
            "pupil_area_ratio_full": 1.35, "pupil_area_ratio_boundary_end": 1.08, "residual_ratio_boundary_end": 9.6, "...": "..."},
  "photo_features": {"sky_mean": 0.61, "...": "..."}
}
```

(위 `pupil` 숫자는 형태를 보여주는 예시다.) 동공은 설명용이라 40세 · τ2 기본(30초)으로 고정했다. 등급은 나이와 상관없다. `speed_kmh`(기본 100)는 동공 칸의 경계부 시간에만 쓴다.

응답에는 화면 어둡기 값 `screen_effect`도 있다: `start_dark_opacity`(하 .30 · 중 .60 · 상 .85), `recover_s` 3, `curve`(0.5초 간격 검은 막 진하기). 화면에 쓰는 법은 `시뮬레이터_요청서_v3.md`를 따른다.

```text
```

## 3. 응답 칸 뜻

| 칸 | 뜻 | 화면에서 쓰는 법 |
| --- | --- | --- |
| `grade` | 입구 순응 위험도 하 · 중 · 상 (대표 모델 결과) | 알림 색: 하 초록 · 중 노랑 · 상 빨강 |
| `grade_prob` | 모델이 각 등급을 얼마나 확신하는지 (합 1) | 화면에 안 씀 (참고값). **사고 확률이 아니다** |
| `est_outside_luminance_cd_m2` | 사진으로 추정한 바깥 밝기 L20 | 화면에 안 씀 (참고값) |
| `tunnel_entrance_luminance_cd_m2` | 지침상 입구 조명 밝기 (280 × 조광 단계) | 화면에 안 씀 (참고값) |
| `est_luminance_ratio` | 바깥 ÷ 터널 안 (참고값) | 화면에 안 씀 (참고값) |
| `warn_distance_m` | 경고를 띄울 거리 (입구 앞 155m, 지침 정지시거) | 차가 입구 155m 전에 오면 알림 (거리 숫자는 화면에 안 띄움) |
| `boundary_zone_m` | 입구 경계부 길이 80m (100km/h에서 약 3초) | 화면에 안 씀 (참고값) |
| `screen_effect` | 입구 통과 순간 검은 막 진하기와 3초 회복 곡선 | 화면 어두워짐 → 3·2·1 → 최소 시야 확보 |
| `pupil.open_fraction_1s` · `_2s` | 들어간 뒤 1초 · 2초에 동공이 필요한 만큼 열린 비율 (약 11% · 19%) | 화면에 안 씀 (참고값) |
| `pupil.pupil_outside_mm` → `pupil_boundary_end_mm` | 바깥에서의 동공 지름 → 경계부 끝(약 3초 뒤) 동공 지름 | 화면에 안 씀 (참고값) |
| `pupil.pupil_area_ratio_boundary_end` | 경계부 끝까지 동공이 더 받아들인 빛 (배) | 화면에 안 씀 (참고값) |
| `pupil.pupil_area_ratio_full` | 동공이 다 따라왔을 때 면적비 | 화면에 안 씀 (참고값) |
| `pupil.residual_ratio_boundary_end` | 휘도비 ÷ 동공 면적비 (망막이 감당할 몫의 산술값) | 화면에 안 씀 (참고값). 시야 손실·사고 확률 아님 |

주의: `grade`는 13개 값을 함께 보고 바로 맞힌 등급이고, `est_luminance_ratio`는 추정 밝기만으로 계산한 참고값이다. 둘의 등급 기준(10배·15배)이 어긋날 수 있다 (시험 사진 입구 66장 중 62% 일치). 화면의 경고는 `grade`를 따른다.

## 4. 모델 요약

- 학습: AI Hub 71786 3,624장 (2023년 8~9월, 카메라 1기종, 입구 1,927 + 터널 없는 고속도로 1,697)
- ① 직선 회귀(Ridge α=10, 사진 숫자 10개 + 해 높이 + 구름) → 바깥 밝기 log10 L20
- ② 추정 밝기 포함 13개 → 2단계 로지스틱 (1단계 하 vs 위험, 2단계 중 vs 상, C=0.01, 등급 수 가중)
- 시험 (처음 보는 카메라 Mapillary 입구 66장): Macro F1 .844, 상 재현율 1.00, 밝기 R² .77
- 한계: 점수 대부분은 해 높이가 설명한다 (해 높이 규칙 .789와 비슷). 학습이 여름 카메라 1기종이라, 겨울 아침처럼 해가 낮은 맑은 날의 '중'을 놓칠 수 있다.

## 5. 파일

| 파일 | 내용 |
| --- | --- |
| `model3_core.py` | 사진 숫자 10개 · 해 높이 · 휘도비 · 등급 식 (07 · 10 노트북과 같음) |
| `train_model.py` | 대표 모델 학습 → joblib 저장, 10 노트북과 대조 (`--test`, `--ref`) |
| `pupil.py` · `동공시간_공식숫자.csv` | 동공 모델 (02 노트북과 같은 식, 01 PsPM 공식 숫자) |
| `api.py` | FastAPI 서버 |
| `00_API_확인.ipynb` | VS Code에서 Run All로 전부 확인 |
| `verify_api.py` | API 결과가 10 노트북과 같은지 확인 |
