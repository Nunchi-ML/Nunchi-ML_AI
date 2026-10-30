<div align="center">

<img src="https://raw.githubusercontent.com/Nunchi-ML/.github/main/profile/assets/logo.png" width="96" alt="Nunchi-ML"/>

# Nunchi-ML_AI

데이터 정제 · EDA · 모델 ① 구간 사고 예측 · 모델 ② 화면 눈부심 분류

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-2E7D32?style=flat-square) ![GeoPandas](https://img.shields.io/badge/GeoPandas-139C5A?style=flat-square) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) ![pvlib](https://img.shields.io/badge/pvlib-F2522E?style=flat-square) ![SHAP](https://img.shields.io/badge/SHAP-FF0D57?style=flat-square) ![Optuna](https://img.shields.io/badge/Optuna-2C5BB4?style=flat-square)

[![Notion](https://img.shields.io/badge/Notion-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%20%ED%99%88-000000?style=flat-square&logo=notion&logoColor=white)](https://app.notion.com/p/40554fa7e143815680c0cfb0ff414d78) [![GitHub](https://img.shields.io/badge/GitHub-Nunchi--ML-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Nunchi-ML)

</div>

---

## 👥 담당

데이터 엔지니어 A · B, AI 엔지니어 A · B

## 📁 폴더

```
📂 data/                원본 · 정제 데이터          ← git에 올리지 않음
📂 notebooks/
 ├─ 📂 eda/             EDA 노트북
 └─ 📂 experiments/     모델 실험 노트북
📂 src/                 같이 쓰는 코드 (불러오기 · 정제 · 분할 · 파이프라인)
📂 models/              저장한 파이프라인 파일      ← git에 올리지 않음 (Release로 전달)
```

## 🧰 기술 스택

| 하는 일 | 기술 |
|---|---|
| 데이터 정리 | pandas · pyarrow |
| 도로 구간 나누기 · 사고 위치 붙이기 | geopandas · shapely (OpenStreetMap 도로) |
| 해 위치(높이 · 방위) 계산 | pvlib |
| 영상 → 프레임, 화면 숫자 뽑기 | OpenCV |
| 모델 ① 구간 사고 건수 | LightGBM (포아송) · 비교 기준선 statsmodels 포아송 회귀 |
| 모델 ② 눈부심 3단계 분류 | LightGBM · scikit-learn · 비교 기준선 밝기 규칙 |
| 검증 | scikit-learn GroupKFold (도로별 · 촬영 날짜별로 나눔) |
| 중요 변수 · 튜닝 | SHAP · Optuna |

> 신경망(딥러닝)은 쓰지 않습니다. 화면은 OpenCV로 숫자를 뽑아 트리 모델에 넣습니다.

## 📏 기준 — 노션이 정본

| | 어디에 | 내용 |
|---|---|---|
| 📏 | 노션 「분석계획서」 | 평가 지표 · fold 수 · 시드 |
| ✂️ | 노션 「데이터 명세서」 | 분할 파일 `folds_v1` 생성 — **모든 실험은 이 파일만 씁니다** |
| 🏷️ | 노션 「데이터 명세서」 | 데이터셋 버전 v1, v2 … 버전이 다르면 점수를 비교하지 않습니다 |
| 🧪 | 노션 「실험 기록」 | 실험 한 건 = 한 줄 (번호 `EXP-`) |

## 📦 모델 넘기기 (AI → BE)

모델②는 LightGBM `model2.joblib` + `meta.json` + 공통 전처리/추론 `nunchi-glare` wheel + 기준 JPEG 3장을 한 묶음으로 전달합니다. pickle에 사용자 정의 클래스를 넣거나 BE에서 특징 코드를 복제하지 않습니다. 전체 프레임을 640×368로 변환하는 기존 학습 방식은 유지합니다.

AI 폴더에서 실행 (로컬 데이터는 git 제외):

```bash
python -m src.glare.export_model --images ../../data/glare --features ../../data/glare/pilot_glare/glare_features.csv --labels ../../data/glare/pilot_glare/pilot_labels.csv --out models/model2-pilot-v1
python -m pip wheel --no-deps --no-build-isolation --wheel-dir models/model2-pilot-v1 .
python -m src.glare.bundle_model models/model2-pilot-v1 models/model2-pilot-v1.zip
```

setuptools>=70이 필요합니다. export는 130장 특징을 원본에서 다시 계산해 CSV와 비교하고, 영상별 GroupKFold 5겹으로 원본·JPEG95 전송 모사·밝기 기준선을 평가한 뒤 전체 시범 라벨로 fit합니다. 점수·영상 분할·라벨 출처와 해시·원본 해상도 분포·라이브러리 버전을 meta.json에 기록합니다. wheel은 export와 같은 코드에서 다시 빌드해야 하며, BE는 모델·특징 해시·런타임 버전을 확인합니다.

시범 재현: 원본 macro F1 0.57135 → JPEG95 0.56477, 심함 재현율 모두 0.51852. Claude 보조 시범 라벨이며 **2인 독립 검증 전**입니다. 브라우저 JPEG·휴대폰·시뮬레이터 합성 화면의 성능을 보장하지 않습니다. 결과 기준 JPEG의 예측은 정답과 다를 수 있습니다.

팀 Release로 전달할 때 모델·wheel·meta·원본 출처/이용 조건을 함께 적고 BE 의존성 버전을 맞춥니다. 실제 Release 발행·배포는 아직 하지 않았습니다. FE는 Vercel, BE는 Render이며 신천대로·달구벌대로 모두 MVP 필수입니다.

`bundle_model`은 모델·전처리 해시와 기준 사진 예측을 확인하고 zip에 manifest(파일별 SHA-256)를 넣습니다. 같은 폴더의 `.zip.sha256`은 BE `MODEL_BUNDLE_SHA256`에 지정합니다. zip을 팀 AI Release에 올리고 해당 태그의 고정 asset URL을 `MODEL_BUNDLE_URL`로 사용합니다. 현재는 로컬 bundle 준비이며 Release 공개 전입니다.

## 🚀 실행

```bash
pip install -r requirements.txt
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
