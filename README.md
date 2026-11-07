<div align="center">

<img src="https://raw.githubusercontent.com/Nunchi-ML/.github/main/profile/assets/logo.png" width="96" alt="Nunchi-ML"/>

# Nunchi-ML_AI

데이터 정제 · 실험 · 태양광 눈부심 분류 · 밝기 전환 부담 · 통합 주행 안내

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-2E7D32?style=flat-square) ![GeoPandas](https://img.shields.io/badge/GeoPandas-139C5A?style=flat-square) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) ![pvlib](https://img.shields.io/badge/pvlib-F2522E?style=flat-square) ![SHAP](https://img.shields.io/badge/SHAP-FF0D57?style=flat-square) ![Optuna](https://img.shields.io/badge/Optuna-2C5BB4?style=flat-square)

[![Notion](https://img.shields.io/badge/Notion-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%20%ED%99%88-000000?style=flat-square&logo=notion&logoColor=white)](https://app.notion.com/p/40554fa7e143815680c0cfb0ff414d78) [![GitHub](https://img.shields.io/badge/GitHub-Nunchi--ML-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Nunchi-ML)

</div>

---

## 👥 담당

이성진(태양광 눈부심 분류) · 최재혁(밝기 전환 부담) · 박범수(통합 주행 안내) · 권수영(데이터·문서)

## 🧭 한눈에 보기

| 모델 | 하는 일 | 상태 | 담당 |
|---|---|---|---|
| ☀️ **태양광 눈부심 분류** (모델②) | 전방 사진 + 해 위치·날씨로 정상·눈부심·심함 3등급 판정 | 발표 모델 선정 (2단계 로지스틱 회귀) · API 시제품 | 이성진 |
| 🚇 **밝기 전환 부담** (모델③) | 터널·지하차도 입구 사진으로 바깥 밝기와 입구 위험도(하·중·상) 추정 | 대표 모델 확정 (추정 밝기 + 2단계 로지스틱) | 최재혁 |
| 🧭 **통합 주행 안내** (모델①) | 두 모델의 결과를 위치·시각·진행 방향에 맞춰 지도·경고로 연결 | 개발 중 (역할·출력 확정 전) | 박범수 |

> 아래 「현재 서비스 목적」부터는 실험 경계와 재현 방법을 자세히 적은 기록입니다. 수치와 판단의 정본은 노션 [모델링 계획서](https://app.notion.com/p/01e2ae67f0e74a8ab5dc8bf3e1576011)입니다.


## 🧭 현재 서비스 목적 · 2026-11-04 사용자 확정

주행 영상과 밝기 변화 정보를 결합해 **전방 정보 식별 저하와 순응 부담을 안내**한다. 기존 사고 예측·관련성 분석은 연구 기록으로 보존하고, 실제 사고 확률 산출은 현재 서비스의 필수 완료 조건에서 제외한다.

| 구성 | 현재 역할 | 출력의 의미 |
|---|---|---|
| 모델① 통합 안내 | ②·③ 결과를 위치·시각·진행 방향에 맞춰 연결 | 지도 구간·현재 경고·사전 주의와 원인. 통합 구현은 후속 작업 |
| 모델② 현재 전방 상태 | 태양광 눈부심으로 사진 속 차선·앞차·표지판 식별이 저하되는 정도 분류 | 사진 기반 0/1/2 판정. 실제 운전자 생리적 시야의 직접 측정과 구분 |
| 모델③ 밝기 전환 부담 | 밝기 차이가 큰 지점에서 순응 부담 추정 | 예상 동공 변화 시각화·주의 안내. 실제 적응시간·사고 위험은 별도 검증 |

통합 규칙안은 두 원인을 각각 표시하면서 더 높은 주의 수준의 경고를 대표로 보여주고 같은 지점의 알림은 묶는 방식이다. 아직 검증된 경고 기준이 아니며, 점수의 임의 가중합을 사고 확률로 표시하지 않는다. 시간·위치·방향·유효시간·계산 버전을 맞추고 결측을 정상/부담0으로 채우지 않는다.

**오프라인 통합 검수기(2026-11-06):** `notebooks/experiments/sun_visible_probe/integration_probe.py`는 모델② 사진 등급과 모델③ 입구 대비 등급을 각각 보존하면서 같은 사진 SHA256/관측ID·구간/촬영시각·위치·방향·수신시각과 모델 버전을 확인한다. 계수 지문은 형식을 검사해 보존하며, 실제 실행 파일 바이트 검증은 각 모델 실행기의 역할이다. 이 검수기 자체는 입력 결과의 실제 추론 출처를 인증하지 않는다. 나이·거리·방향 허용값은 호출자가 명시하는 미검증 시험 정책이다. 기존 .5 판정은 그대로 보존한다(②는 >=.5, ③ 노트북 sklearn.predict는 정확한 동률에서 False이므로 >.5). `model3`는 `ridge_estimated_luminance_two_stage`의 입구 등급만 받으며 이전 밝기 공식·동공 지름·체험 효과·출구 결과로 대체하지 않는다.

검수기는 실제 학습·서빙/웹 연결을 하지 않는다. 두 결과가 모두 정상일 때만 `observed_normal`이며, 한쪽 정상+다른쪽 미확인은 등급 공란이다. 한쪽 주의+다른쪽 미확인은 알려진 주의만 하한으로 유지한다. 미래 ETA는 현재 관측과 분리해 등급을 비워 둔다. `policy_validated=false`, 사고 확률은 공란이다. 모델③ 학습표·코드는 후속으로 수신해 아래 추론 묶음까지 만들었지만, 같은 실제 입구 사진의 두 모델 결과·외부 시험 대조 전에는 실제 통합 완료로 세지 않는다. 합성 계약 테스트는 모델 정확도/운전자 효과 검증과 구분한다. 기존 운영 모델/API 변경은 없다.

비공개 사례 JSON 목록을 `python notebooks/experiments/sun_visible_probe/integration_probe.py --input data/<cases.json> --out data/<새폴더>`로 검사한다. 각 사례는 `case_id`, `evidence`, `query`, `model2`, `model3`, `policy`를 갖는다. query는 관측ID/구간ID/사진SHA256·위경도/방향·촬영/조회/ETA시각·zone, 각 모델은 공통 context(조회·ETA 대신 received_at), 버전/variant/target·계수SHA256·원등급/A/B/joint 점수를 받는다. 모델③은 입구/추정 휘도(cd/m²)를 명시한다. 출력 폴더는 새로 만들며 사진·라벨·개별 결과/모델 계수는 공개하지 않는다. 예시 계약은 `tests/test_integration_probe.py`의 합성 fixture를 참고한다.

**모델③ 고정 내보내기:** `model3_export_probe.py`는 수신한 학습표3,624행(입구1,927·터널없는도로1,697/장면2,433)과 대표 노트북의 동일한 학습 정의로 Ridge α=10·C=.01을 내보낸다. 학습 쪽 밝기는 장면5겹 OOF이며 추가 튜닝/등급변경은 없다. 필요한 함수/상수 선언만 읽고 노트북의 탐색·파일입출력 셀은 실행하지 않는다. 추가 구조 비교가 붙은 수신 노트북과 저장소 대표 모델의 학습 선언이 같음을 먼저 확인한다. 3,624행에서 원래 노트북/JSON 재로딩의 등급 전부일치·밝기로그 최대오차0·A/B 최대오차1.67e-16을 확인했다. 이는 학습행의 내보내기 동등성 검사이며 외부F1 .844 재현/새성능향상이 아니다. 시험표118행/입구66결과는 아직 수신하지 않았다.

AI 폴더에서 `python notebooks/experiments/sun_visible_probe/model3_export_probe.py --train data/<학습표.csv> --source-notebook data/<전달된노트북.ipynb> --out data/<새묶음>`으로 생성한다. JSON계수·메타·원래 사진10특징/NOAA해높이 전처리만 비공개 묶음에 둔다. `Model3Candidate(directory).predict_features(frame, zone='entry')` 또는 `predict_image(path, {lat,lon,cloud,captured_at}, zone='entry')`를 사용한다. 시각은 시간대가 필요하며 실제 입구로 확인한 사진만 입력한다. 모델②의 pvlib/상대특징으로 모델③ 전처리를 바꾸거나 JPEG를 물리적 조도로 간주하지 않는다. 중앙값 대체는 원래 모델의 학습 중앙값이고 대체된 특징 수를 반환한다. 입력/계수·소스 지문·특징 순서·입구 범위와 원본/재로딩 동등성은 테스트로 확인한다. CSV·원본/모델 계수·전체 결과는 GitHub에 넣지 않는다.

**같은 사진의 오프라인 공동 실행기:** `joint_candidate_probe.py`는 고정②/③ 묶음을 한 JPEG와 해당 사진의 실제 촬영정보에 적용한 뒤 통합 검수기로 전달한다. 모델②는 photo_g+상대특징, 모델③은 원래 사진10특징/NOAA해높이로 각각 계산하며 원출력을 보존한다. 입구로 확인한 근거·관측/구간ID·시험정책·evidence를 명시해야 한다. 사진이 두 실행 사이에 바뀌면 거절하고 실제 수신/조회시각을 사용하므로 과거 사진을 오늘 재실행해도 신선한 관측으로 바뀌지 않는다. 합성 입력은 두 모델의 구동검증에만 쓰고 실제 안내는 미확인으로 남긴다. 모델계수 묶음은 별도 비공개 자료이며 새 웹 API를 추가하지 않는다. `--help`의 필수 인수로 실행하고 새 결과는 AI/data 안에만 저장한다. 실제 동일 입구사진·완전한 G/기상정보·외부66결과의 검증은 별도다.

### 발표·보고서와 구현 상태 대조 (2026-11-07)

**과학·신체 원리(설계 배경):** 눈 안의 산란광이 망막상의 대비와 시인성을 낮출 수 있다는 [IES의 disability glare 정의](https://ies.org/definitions/disability-glare/), 밝기 변화에 따른 [망막의 감도 적응(Webvision)](https://www.webvision.pitt.edu/book/part-viii-psychophysics-of-vision/light-and-dark-adaptation/)을 두 문제의 배경으로 설명한다. [동공은 들어오는 빛의 양을 조절한다(NEI)](https://www.nei.nih.gov/eye-health-information/healthy-vision/how-eyes-work). 현재 사진 등급·입구 대비 등급과 실제 운전자의 시야·적응 완료시간 측정은 구분한다. 동공 크기만으로 적응 완료를 판단하거나 카메라 과노출/3D 번짐을 눈 안 산란의 측정값으로 보지 않는다. 발표는 원리45초+모델 연결15초의1장, 세부 수식/완전 암순응 곡선은 부록으로 둔다. 출처 확인2026-11-06.

| 발표 이름 | 현재 확인한 범위 | 구분할 내용 |
|---|---|---|
| 태양광 눈부심 분류 모델(②) | B_geometry_aug 발표 시제품 기준 선택. 반복 촬영CV F1 .63965·심함 재현율 .64110, 정상→심함 회차평균16.63→7.87 | 원자료 조건부 .63568/.69406과 교환. 개발 중 평가자료 반복 참조·F1 차이 구간0 포함. 독립 우위/운영 교체 아님 |
| 밝기 전환 부담 모델(③) | 확인된 입구 사진의 Ridge α=10 추정 휘도 + balanced 2단계 로지스틱 C=.01. 학습 쪽 휘도는 장면5겹 OOF | 입구66장 F1 .844는 팀 결과. 로컬 학습3,624행 내보내기 동등성과 외부결과 재현을 구분. 출구·동공·적응시간 검증 아님 |
| 통합 주행 안내 모델(① 역할) | 원인별 결과 보존·사진/시각/위치/방향/결측 검수·같은 JPEG 공동 실행. PR66의430테스트·CI 성공 | 실제 계수+명시적 합성 입력의 구동 검증이며 실제 입구 공동 정확도 아님. 두 점수를 사고 확률로 합산하거나 독립 확신도로 취급하지 않음 |

선정 모델②는 운영 `/predict/glare/photo-g`(BE PR6)와 시뮬레이터 카메라/대표3코스 저장 판정(FE PR17)에 연결됐고, 휴대폰 센서 입력 경로(FE PR19)도 구현됐다. 휴대폰 화각66°·대구 편각−8.5°는 가정이고 날씨는 사용자 선택이며 실제 휴대폰 현장 시험은 미완료다. 사진전용 `/predict/glare`는 시범/센서 미확보 시 대체용으로 유지한다. 이 단독 모델 연결, 별도 `/photo-g/` 로컬 분석, PR66의②③ 오프라인 공동 실행은 다른 완료 범위다. 실제 도로변 관측·대안 경로는 비어 있지만 `route_analysis.json`에는3코스43지점의 시뮬레이션 사진 모델② 결과가 있다. 모델③/실제 도로변/공동 결과의 운영 통합과 효용 검증은 남았다. 출구 회복·스마트 유리는 체험 계산이다. 도시CCTV·실교통/우회·전광판/차량 제어·운전자 효과는 확장 검증이며 사진+촬영정보를 ‘경로·시각만 사용/장비 불필요’로 설명하지 않는다.

모델③ 후속 [EXP-94 구조 교차](https://github.com/Nunchi-ML/Nunchi-ML_AI/pull/63)와 [대표 확정 상세](https://github.com/Nunchi-ML/Nunchi-ML_AI/pull/64), [EXP-95 확대 평가](https://app.notion.com/p/41054fa7e1438118b03fe5dc09656e38)는 팀 기록으로 인용한다. EXP-95의 일반도로504장은 ‘입구가 있다면’의 가정 등급이고 실제 입구는 여전히66장이다. 합친570장 F1 .745를 입구66장 .844와 같은 시험의 성능 변화로 연결하지 않는다. 확대 평가의 해높이 규칙 대비 관측 이득과 기존 부스팅 대비 한계를 함께 보고하며, 로컬 외부결과 대조나 서비스 적용 완료로 세지 않는다. [최종 보고서의 현재 상태](https://app.notion.com/p/94cdda5ba9344f859dafdb428c1c8424)를 함께 확인한다.

#### ML이 규칙보다 나았던 조건과 근거

규칙에도 학습 자료로 경계를 고를 기회를 주고, 같은 비교 안의 자료·정답·분할/지표를 맞춘다. ML−규칙의 촬영/장면 군집 재표집95% 구간이0보다 높은 다음 비교를 발표의 핵심 근거로 제시한다.

| 비교 | ML / 규칙 | 재표집 평균 차이 [95% 구간] | 해석 범위 |
|---|---|---|---|
| 실험24: 원320장 학습→재사용 AI Hub, 당시2단계 ML vs 경계학습 규칙, 눈부심 여부 정확도 | .832 / .457 | +.393 [.123,.693] | 해당 개발/스트레스 자료의 알림정확도 이득. 선정 B_geometry_aug의 독립 외부시험 아님 |
| EXP-94: 입구66장, 같은 추정밝기+2단계의 ML 조합 vs 물리밝기+2단계if, Macro F1 | .844 / .581 | +.266 [.072,.424] | 이 상위 구조의 조합 비교. 밝기 추정 하나만의 효과를 분리한 것은 아님 |
| EXP-95: 입구66+가정 도로504=570장, 고정 대표 ML vs 해높이 경계 규칙, Macro F1 | .745 / .581 | +.164 [.093,.226] | 이 확대 평가의 해높이 규칙 대비 이득. 실제 입구는66장 |

출처는 [실험24/PR19](https://github.com/Nunchi-ML/Nunchi-ML_AI/pull/19)·위 EXP-94/95 원기록이다. 차이값은 각 원코드의 재표집 평균으로 원점수의 단순 차이와 다를 수 있다. 원평가 결과를 다시 구성했으며 새 학습·외부 재현을 한 것은 아니다. 눈부심3등급 F1 .63965 vs 원튜닝규칙 .63568과 순응 입구66장 최고 해높이규칙 .789 대비 우위는 미확정이다. 확대570장의 부스팅 .767·물리식 .736 대비 우위도 확정되지 않았다. 평가집합·지표가 다른 행끼리 순위를 비교하거나 모든 조건/규칙에 대한 우위로 확대하지 않으며 기존 F1 주지표와 반복개발·Ridge 선택 한계를 유지한다.

모델②는 기존 정답과 성진50장 원본 라벨을 보존한다. 동일320장·촬영그룹 CV의 기준 Macro F1 .6229와 새 특징을 비교하며, 성진50장 사진 모델 평가는 다른 조건으로 보고한다. 2026-11-06 발표 기준 모델을 `B_geometry_aug`(2단계 로지스틱, 2단계에 추가 자료·각도 정보)로 선정했다. 운영 API 교체와 독립 일반화 검증은 별도다. 사진의 대비나 밝기 수식으로 정답을 새로 만들어 같은 입력으로 평가하지 않는다.

**세 AI에 라벨링을 맡긴 목적:** 사람이 사진만 보고 정하는 등급의 주관·경계 불확실성을 보완하려고 GPT·Claude·Gemini에 같은 기준으로 판정을 맡겼다. 사람 정답이 정확하다고 전제한 AI 평가가 아니라, 한 판정자에 대한 의존을 줄이고 합의·불일치·기권과 근거를 비교해 라벨 기준을 점검하는 시도다. 같은50장의 교차판정과 SD 개발300장의 세AI 라벨링은 별도 자료다. SD300의 2026-11-05 기록은 유효560/예정900판정·한도 부분완료, 공통110장 중3자일치101·불일치7·기권포함2다. 정상 편중·심함 희소가 있어 일치율을 정확도/최종정답 신뢰도로 보지 않는다. 사람/각AI 원답은 별도 보존했고, 라벨별 고정예측 재채점·GPT 참고등급 평가와 AI 라벨 재학습에 의한 개선을 구분한다. AI가 사람보다 정확하다는 검증이나 AI 합의의 실제 운전자 정답 검증은 아니다. 최종 분류기는 전통 머신러닝이며 AI 이미지 판정 도구와 구분한다.

동공 크기 추정은 밝기에 적응한 상태의 예상 크기이며 적응시간을 직접 뜻하지 않는다([Watson·Yellott, 2012](https://pubmed.ncbi.nlm.nih.gov/23012448/)). 보정되지 않은 JPEG 픽셀을 물리적 휘도로 간주하지 않는다. 상세 역할·후속 상태 정본은 [모델링 계획서](https://app.notion.com/p/01e2ae67f0e74a8ab5dc8bf3e1576011)의 「현재 역할과 통합 원칙」이다. 아래 기존 사고 모델 설명은 연구·코드 구조 기록으로 보존한다.

## 📁 폴더 — 어디에 뭘 두나

```
📂 data/                          원본 · 정제 데이터              ← git에 안 올림 (팀 구글 드라이브에서 받기)
📂 notebooks/                     눈으로 보며 하는 작업 (노트북)
 ├─ 📂 eda/                       데이터 둘러보기
 └─ 📂 experiments/               모델 실험 — 실험 하나 = 폴더 하나
     ├─ 📂 glare_pilot/           모델 ② 눈부심 시범 검증
     └─ 📂 tunnel_adaptation/     모델 ③ 터널 순응 부담 (01~06 노트북)
📂 src/nunchi_ml/                 같이 쓰는 코드 = 파이썬 패키지 nunchi_ml (BE도 이걸 그대로 씀)
 ├─ 📂 glare/                     모델 ② 전처리(features) · 추론(inference) · 배포 묶음 만들기
 ├─ 📂 tunnel/                    모델 ③ 점수 A 계산식
 └─ 📂 accident/                  모델 ① 사고 위험 (만들 때 여기에)
📂 tests/                         nunchi_ml 테스트 — PR마다 자동으로 돌아감 (CI)
📂 models/                        만든 모델 파일 · 배포 묶음     ← git에 안 올림 (GitHub Release로 전달)
```

### 이렇게 쓰면 돼요

| 하고 싶은 것 | 어디에 |
|---|---|
| 데이터 처음 살펴보기 | `notebooks/eda/` 에 노트북 |
| 모델 실험 | `notebooks/experiments/<실험이름>/` 에 노트북 + `README.md` (뭘 했는지 · 결과) |
| 여러 노트북이 같이 쓰는 함수, **서버(BE)에서도 쓰는 코드** | `src/nunchi_ml/<모델>/` 에 `.py` |
| 새 모델 시작 (예: 모델 ①) | `src/nunchi_ml/accident/` 폴더 + 빈 `__init__.py` 만들기 |

- 노트북에서 불러오기: 처음 한 번 `pip install -e .` 한 뒤 `from nunchi_ml.tunnel.adaptation import adaptation_scores` 처럼 (경로 장난 `sys.path` 필요 없음)
- 명령줄 실행: `python -m nunchi_ml.tunnel.adaptation 3.0`

### 원본 데이터는 git에 안 올려요

- 용량이 크고(영상 · 이미지 · 수만 행 CSV), 일부는 이용 조건상 다시 배포하면 안 돼요
- `.gitignore`가 `data/` 안의 파일과 `*.csv` · `*.parquet` · `*.xlsx` · `*.joblib` 등을 막아 둬서 실수로 올라가지 않아요
- 원본은 **팀 구글 드라이브**에서 받아 각자 `data/` 에 두고, 폴더 구조 · 파일 이름은 노션 「데이터 명세서」를 따라요
- 만든 모델은 git 대신 **GitHub Release**로 넘겨요 (아래 「모델 넘기기」)

## 🧰 기술 스택

| 하는 일 | 기술 |
|---|---|
| 데이터 정리 | pandas · pyarrow |
| 도로 구간 나누기 · 사고 위치 붙이기 | geopandas · shapely (OpenStreetMap 도로) |
| 해 위치(높이 · 방위) 계산 | pvlib |
| 영상 → 프레임, 화면 숫자 뽑기 | OpenCV |
| 모델 ① 구간 사고 건수 | LightGBM (포아송) · 비교 기준선 statsmodels 포아송 회귀 |
| 모델 ② 눈부심 3단계 분류 | LightGBM · scikit-learn · 비교 기준선 밝기 규칙 |
| 모델 ③ 터널 순응 부담 | statsmodels 조건부 로지스틱 (같은 자리 비교) |
| 검증 | scikit-learn GroupKFold (도로별 · 촬영 날짜별로 나눔) |
| 중요 변수 · 튜닝 | SHAP · Optuna |

## 📏 기준 — 노션이 정본

| | 어디에 | 내용 |
|---|---|---|
| 📏 | 노션 「분석계획서」 | 평가 지표 · fold 수 · 시드 |
| ✂️ | 노션 「데이터 명세서」 | 분할 파일 `folds_v1` 생성 — **모든 실험은 이 파일만 씁니다** |
| 🏷️ | 노션 「데이터 명세서」 | 데이터셋 버전 v1, v2 … 버전이 다르면 점수를 비교하지 않습니다 |
| 🧪 | 노션 「실험 기록」 | 실험 한 건 = 한 줄 (번호 `EXP-`) |

## 📦 모델 넘기기 (AI → BE)

모델②는 LightGBM `model2.joblib` + `meta.json` + 공통 코드 `nunchi-ml` wheel + 기준 JPEG 3장을 한 묶음으로 전달합니다. (첫 묶음 `model2-pilot-v1`은 이전 이름 `nunchi-glare` wheel로 나갔고, BE는 두 이름 모두 받습니다)

> ⚠️ `src/nunchi_ml/glare/features.py`는 **한 글자도 고치지 마세요** — 배포된 모델이 이 파일의 지문(sha256)을 확인해서, 바뀌면 서버가 모델을 못 불러와요. 고쳐야 하면 모델을 다시 export하고 `tests/test_glare.py`의 지문 값도 같이 바꿉니다. 줄바꿈도 지문에 들어가서 `.gitattributes`로 CRLF 고정해 뒀어요. pickle에 사용자 정의 클래스를 넣거나 BE에서 특징 코드를 복제하지 않습니다. 전체 프레임을 640×368로 변환하는 기존 학습 방식은 유지합니다.

AI 폴더에서 실행 (로컬 데이터는 git 제외):

```bash
pip install -e .
python -m nunchi_ml.glare.export_model --images ../../data/glare --features ../../data/glare/pilot_glare/glare_features.csv --labels ../../data/glare/pilot_glare/pilot_labels.csv --out models/model2-pilot-v1
python -m pip wheel --no-deps --no-build-isolation --wheel-dir models/model2-pilot-v1 .
python -m nunchi_ml.glare.bundle_model models/model2-pilot-v1 models/model2-pilot-v1.zip
```

setuptools>=70이 필요합니다. export는 130장 특징을 원본에서 다시 계산해 CSV와 비교하고, 영상별 GroupKFold 5겹으로 원본·JPEG95 전송 모사·밝기 기준선을 평가한 뒤 전체 시범 라벨로 fit합니다. 점수·영상 분할·라벨 출처와 해시·원본 해상도 분포·라이브러리 버전을 meta.json에 기록합니다. wheel은 export와 같은 코드에서 다시 빌드해야 하며, BE는 모델·특징 해시·런타임 버전을 확인합니다.

시범 재현: 원본 macro F1 0.57135 → JPEG95 0.56477, 심함 재현율 모두 0.51852. Claude 보조 시범 라벨이며 **2인 독립 검증 전**입니다. 브라우저 JPEG·휴대폰·시뮬레이터 합성 화면의 성능을 보장하지 않습니다. 결과 기준 JPEG의 예측은 정답과 다를 수 있습니다.

팀 Release로 전달할 때 모델·wheel·meta·원본 출처/이용 조건을 함께 적고 BE 의존성 버전을 맞춥니다. 실제 Release 발행·배포는 아직 하지 않았습니다. FE는 Vercel, BE는 Render이며 신천대로·달구벌대로 모두 MVP 필수입니다.

`bundle_model`은 모델·전처리 해시와 기준 사진 예측을 확인하고 zip에 manifest(파일별 SHA-256)를 넣습니다. 같은 폴더의 `.zip.sha256`은 BE `MODEL_BUNDLE_SHA256`에 지정합니다. zip을 팀 AI Release에 올리고 해당 태그의 고정 asset URL을 `MODEL_BUNDLE_URL`로 사용합니다. 현재는 로컬 bundle 준비이며 Release 공개 전입니다.

## 🚀 실행

```bash
pip install -r requirements.txt   # 노트북용 라이브러리 전부
pip install -e ".[test]"          # 공통 코드 nunchi_ml 을 내 컴퓨터에 연결 (고치면 바로 반영)
pytest -q                          # 공통 코드 테스트 (PR마다 GitHub에서도 자동으로 돌아감)
```


<details>
<summary><b>🤝 작업 규칙</b> (펼치기)</summary>

<br/>

| 항목 | 규칙 |
|---|---|
| 🌿 브랜치 | `feat/` 기능 · 실험 · `fix/` 버그 · `docs/` 문서 · `refactor/` 정리 · `data/` 데이터 처리 + 짧은 설명 (예: `feat/lgbm-baseline`) |
| ✍️ 커밋 | `feat: 베이스라인 모델 추가`처럼 접두어 + 한 줄 |
| 🔀 병합 | `main`에 바로 올리지 않고 PR → squash 병합 |
| 💬 공유 | 착수 전 슬랙 `[작업 시작]`, 병합 후 `[작업 정리]` + 노션 카드에 PR 링크 |
| 🔍 리뷰 | 코멘트 등급 P1 꼭 고침 · P2 고치면 좋음 · P3 의견 |
| 🏷️ 라벨 | 브랜치 이름으로 자동으로 붙음 |

📘 파일 이름 규칙 · 올리면 안 되는 것 · 설치 없이 웹에서 올리는 법: [노션 「GitHub 컨벤션」](https://app.notion.com/p/c86986ef9f3a4a3caedabfc16b6ff0b7)

</details>
