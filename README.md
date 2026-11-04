<div align="center">

<img src="https://raw.githubusercontent.com/Nunchi-ML/.github/main/profile/assets/logo.png" width="96" alt="Nunchi-ML"/>

# Nunchi-ML_AI

데이터 정제 · EDA · 모델 ① 통합 안내 · 모델 ② 전방 식별 저하 분류 · 모델 ③ 밝기 전환 순응 부담

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-2E7D32?style=flat-square) ![GeoPandas](https://img.shields.io/badge/GeoPandas-139C5A?style=flat-square) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) ![pvlib](https://img.shields.io/badge/pvlib-F2522E?style=flat-square) ![SHAP](https://img.shields.io/badge/SHAP-FF0D57?style=flat-square) ![Optuna](https://img.shields.io/badge/Optuna-2C5BB4?style=flat-square)

[![Notion](https://img.shields.io/badge/Notion-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%20%ED%99%88-000000?style=flat-square&logo=notion&logoColor=white)](https://app.notion.com/p/40554fa7e143815680c0cfb0ff414d78) [![GitHub](https://img.shields.io/badge/GitHub-Nunchi--ML-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Nunchi-ML)

</div>

---

## 👥 담당

데이터 엔지니어 A · B, AI 엔지니어 A · B

## 🧭 현재 서비스 목적 · 2026-11-04 사용자 확정

주행 영상과 밝기 변화 정보를 결합해 **전방 정보 식별 저하와 순응 부담을 안내**한다. 기존 사고 예측·관련성 분석은 연구 기록으로 보존하고, 실제 사고 확률 산출은 현재 서비스의 필수 완료 조건에서 제외한다.

| 구성 | 현재 역할 | 출력의 의미 |
|---|---|---|
| 모델① 통합 안내 | ②·③ 결과를 위치·시각·진행 방향에 맞춰 연결 | 지도 구간·현재 경고·사전 주의와 원인. 통합 구현은 후속 작업 |
| 모델② 현재 전방 상태 | 태양광 눈부심으로 사진 속 차선·앞차·표지판 식별이 저하되는 정도 분류 | 사진 기반 0/1/2 판정. 실제 운전자 생리적 시야의 직접 측정과 구분 |
| 모델③ 밝기 전환 부담 | 밝기 차이가 큰 지점에서 순응 부담 추정 | 예상 동공 변화 시각화·주의 안내. 실제 적응시간·사고 위험은 별도 검증 |

통합 규칙안은 두 원인을 각각 표시하면서 더 높은 주의 수준의 경고를 대표로 보여주고 같은 지점의 알림은 묶는 방식이다. 아직 검증된 경고 기준이 아니며, 점수의 임의 가중합을 사고 확률로 표시하지 않는다. 시간·위치·방향·유효시간·계산 버전을 맞추고 결측을 정상/부담0으로 채우지 않는다. 이번 변경은 문서 기준이며 운영 모델·API 변경은 없다.

모델②는 기존 정답과 성진50장 원본 라벨을 보존한다. 동일320장·촬영그룹 CV의 기준 Macro F1 .6229와 새 특징을 비교하며, 성진50장 사진 모델 평가는 다른 조건으로 보고한다. 성능 개선이 검증될 때까지 최종 모델 결정을 보류한다. 사진의 대비나 밝기 수식으로 정답을 새로 만들어 같은 입력으로 평가하지 않는다.

**AI 보조 라벨링은 검토 제안이다.** 기존 Claude 임시 라벨은 사람과 공통48장 중34장 일치(70.8%, 카파 .473)했다. Gemini 등 추가 이미지 이해 모델 후보에는 사람·기존 AI 답안을 숨기고 태양/번짐 근거, 대상별 식별성, 저하 원인, 등급 또는 판단 보류를 기록하게 하는 방안을 검토한다. 후보 우위·새 AI 판정은 미검증이며 AI 합의를 실제 운전자 정답으로 간주하지 않는다. AI 학습 라벨과 사람 평가를 구분한다. 최종 분류기는 딥러닝·신경망 임베딩 없이 전통 머신러닝으로 실험한다.

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
