# 사진 상대 특징 두 단계 (후보 비교) — 모델 ②

**상태: 후보 비교만 한다. 최종 모델(G 만) · 배포 features.py · 모델/API 는 바꾸지 않는다.**

화면 숫자 12개(`glare/features.py`)는 절대 밝기 문턱을 써서 카메라가 바뀌면 무너졌다. 여기서는 사진 한 장 안에서만, 그 사진의 중앙값 · 최고 밝기에 대한 **비율**로 계산한 특징 12개를 만들고, 눈부심 여부 쪽에만 붙인 **두 단계**(여부 = G + 상대 12, 심함 = G)를 같은 프로토콜로 비교한다.

| 파일 | 역할 |
|---|---|
| `rel_features.py` | 상대 특징 12개 (`peak_med · dyn · blob_area · blob_circ · sat_top · ring1~3 · glow_con · disc_sat · peak_x/y`). 라벨 · 해 위치 · 카메라 통계를 쓰지 않는다. |
| `sun_visible_probe.py` | 624장에서 G 만 / 두 단계(기존 화면 12 · 상대 12) / 사진 특징 단독을 세 가지 검증으로 비교 (초기 탐색용 — 판정은 3단계 argmax 를 쓰므로 아래 프로토콜 결과를 기준으로 본다) |
| `ablation.py` | 후진 제거로 12 → 3개. 선택은 세 카메라 점수만, 재사용 AI Hub 는 마지막 채점만. |
| `codex_protocol_compare.py` | `g_visibility_probe.py`(PR #16) 프로토콜(고정 스냅샷 · previous 날씨 · 같은 판정 함수 · 촬영 CV 5×30 · 카메라 제외 · 재사용 AI Hub)에 두 단계 구조와 상대 12 행을 더해 비교. Codex 파일은 import 만 하고 수정하지 않는다. |
| `match_probe.py` | **(기각)** G 가 계산한 해의 화면 위치와 사진의 가장 밝은 점의 거리(`sun_dist · sun_match`)를 특징으로 더한다. |
| `synth_probe.py` | **(기각)** 정상 사진에 해 원반 · 번짐 · 베일을 합성해 눈부심 · 심함 예시를 늘려 학습한다. |
| `new_camera_eval.py` · `new_camera_audit.py` | 새 영상 시험. AI 임시/사람 1인/사람 합의를 구분하고 상수 기준선·균형 정확도·영상별 결과·입력/소스 해시를 기록한다. |
| `prepare_human_review.py` | 기존 사진 ID를 유지한 50장 블라인드 검수 표본을 복사한다. 기존 라벨 묶음/답안은 덮어쓰지 않는다. |

## 기각한 후보 (같은 Codex 프로토콜)
| 후보 | 핵심 결과 | 왜 접었나 |
|---|---|---|
| 해 위치-사진 일치도 | 세 카메라에서는 눈부심일 때 더 가깝지만 **AI Hub 는 구분 안 됨**(.43 vs .47). 일치도만 쓰면 AI Hub 알림재현율 .38 | 진행 방향 근사 · 지평선 가정으로 예상 해 위치 자체가 틀린다. 상대 12 에 더하면 광각은 좋아지나(.835→.866) AI Hub 알림정확도 .83→.80 |
| 합성 글레어 증강 | AI Hub 알림재현율 .88→.56, 블랙박스 제외 헛알림 61→97%, 심함까지 이미지로 판정하면 CV 심함 재현율 .66→.003 | 합성 심함은 임의 정의라 실제와 다르다. 노출만 흔든 대조도 AI Hub 헛알림을 28→45% 로 악화 |

## 실행 (재현)
```
python notebooks/experiments/sun_visible_probe/rel_features.py        # (모듈 — 직접 실행하지 않음)
python notebooks/experiments/sun_visible_probe/sun_visible_probe.py   # 첫 실행에 data/sun_visible/rel_features.csv 생성
python notebooks/experiments/sun_visible_probe/ablation.py
python notebooks/experiments/sun_visible_probe/codex_protocol_compare.py \
    --snapshot data/analysis_20261104/model2_snapshot.csv \
    --rel data/sun_visible/rel_features.csv \
    --cache <g_visibility_probe 가 만든 image_features.csv> --out <결과 폴더>
```
사진 · 특징 CSV · 결과는 git 에 올리지 않는다 (`data/`).

## 결과 요약 (코드 기준 main 6062b08, Codex 프로토콜)
| 입력 | 촬영CV F1 | CV 정상헛알림 | 서울 폰 제외 알림정확도 | 광각 제외 알림정확도 | 재사용AIHub 알림정확도 · 헛알림 · 알림재현율 |
|---|---|---|---|---|---|
| G | .543 | 52% | .63 | .86 | .74 · 83.0% · 1.00 |
| 한 모델 G+밝은덩어리6 (PR #16) | .565 | 34% | .85 | .83 | .74 · 27.7% · .75 |
| **두 단계 G+상대12** | **.623** | 33% | .87 | .84 | **.83** · 27.7% · .88 |
| 한 모델 G+상대12 | .511 | 33% | .87 | .84 | .83 · 27.7% · .88 |

- AI Hub 정상 중 헛알림 감소(83% → 27.7%)는 주행 33개 군집 부트스트랩 95% 구간 [−69%p, −33%p] 로 0 을 포함하지 않는다.
  알림정확도 +.10 [−.01, +.31], 알림재현율 −.11 [−.22, .00] 은 구간이 0 에 걸려 확정하지 못한다.
- 두 단계의 효과는 **심함 보존**이다 (한 모델 CV 심함 재현율 .13 → 두 단계 .66). 알림 지표는 한 모델과 같다.
- 12개 중 하나씩 빼면 점수가 가장 떨어지는 것: `disc_sat · blob_circ · ring3`. 4~6개로 줄이면 세 카메라 점수는 오르지만 재사용 AI Hub 알림정확도가 .83 → .74~.75 로 떨어져 줄이지 않았다.

## 새 영상 시험과 사람 검수 — 2026-11-04

기존 `data/new_cameras/label_pack/` 349장은 재생성하지 않는다. `new_camera_frames.py`는 기존 폴더가 있으면 중단한다. 라벨 도구는 묶음의 HTML을 열면 사진이 바로 보이며, 이름 입력 → 판정자 적용 후 기록한다. 답안 저장은 사진 묶음·이름별로 분리한다. CSV는 0/1/2, X는 제외다. 해/번짐 없이 하늘만 하얀 사진은 0, 태양광 때문에 주요 도로 정보가 현저히 식별되지 않을 때 2다. 기존 등급 라벨을 자동 변경하지 않는다.

```powershell
# 새 표본을 처음 준비할 때만 (기존 폴더가 있으면 중단)
python notebooks/experiments/sun_visible_probe/prepare_human_review.py --root data/new_cameras --out data/new_cameras/human_review_50

# AI 임시 결과 재현: 사람 검증 성적으로 보고하지 않음
python notebooks/experiments/sun_visible_probe/new_camera_eval.py --pack data/new_cameras --labels data/new_cameras/labels_claude.csv --label-kind ai_provisional --snapshot data/analysis_20261104/model2_snapshot.csv --rel data/sun_visible/rel_features.csv --cache data/analysis_20261104/step3_visibility/image_features.csv --out data/new_cameras/audit_20261104

# 사람 CSV가 모인 뒤: 동명이인·동일 파일은 독립 판정자로 세지 않음
python notebooks/experiments/sun_visible_probe/new_camera_eval.py --pack data/new_cameras --labels data/new_cameras/labels_a.csv data/new_cameras/labels_b.csv --label-kind human_consensus --snapshot data/analysis_20261104/model2_snapshot.csv --rel data/sun_visible/rel_features.csv --cache data/analysis_20261104/step3_visibility/image_features.csv --out data/new_cameras/human_evaluation
```

`human_review_50/`은 AI 라벨 일치도 확인용 표본으로 전체 시험의 사람 2인 검증을 대신하지 않는다. 이 50장과 AI 라벨의 카파를 계산할 때 AI를 사람 합의 정답에 섞지 않는다. 사진·라벨·예측·검수 ZIP은 `data/`에만 두고 공개 GitHub에는 올리지 않는다.

50장 사람 1인 CSV를 평가할 때는 `--label-kind human_single`을 쓰며, `--compare-ai data/new_cameras/labels_claude.csv`를 더하면 AI와의 일치도를 **별도로** 기록한다. AI 제외 행은 사람 평가에서 삭제하지 않고 일치도 비교에서만 비교불가로 남긴다. `human_ai_comparison.csv`와 `report.json`의 `human_ai_agreement`에서 일치율·비가중/선형/이차 카파·혼동행렬·비교 불가 수를 확인한다. CSV의 `labeler`는 판정자 자기기입 정보이며, 파일 검증만으로 실제 독립 판정을 입증하지는 않는다.

사람 1인 50장(14영상)의 첫 확인: 정상32·눈부심8·심함10. AI가 X로 제외한 2장 외 48장 일치율 .708, 코헨 카파 .473(선형 .594, 이차 .722), 사람 심함10장 중 AI는7장을 눈부심으로 낮게 판정했다. 같은 학습624장 모델을 사람50장에 평가하면 LightGBM 정확도 .540·균형정확도 .531·F1 .318·심함재현율 .200, 상대12 로지스틱 .540·.568·.213·.000, 최선단일특징 규칙 정확도 .580, 전부정상 .640이다. ML−최선규칙 정확도 차이는 둘 다 −.04이고 영상 군집95%구간은0을 포함한다. 사람1인·50장 탐색이며 일반화/ML 우위를 확정하지 않는다. 사람 파일은 보존하고 AI 라벨을 자동 덮어쓰지 않는다.

기존 AI 라벨 시험 325장/13영상 결과를 재현했다. LightGBM 알림정확도 .603·균형정확도 .572·재현율 .513, 상대12 로지스틱 .498·.508·.526이다. **전부 정상 기준선은 정확도 .766·균형정확도 .500·재현율 0**이다. 따라서 밝기 규칙 대비 정확도 차이만으로 실사용 성능이나 일반화 성공을 선언하지 않는다. ML1과 최선 단일 특징 규칙의 차이는 확정되지 않았고, 이 시험에서 상대12가 기존 화면12보다 낫다는 근거도 없다. AI 라벨, 장면 편중, 영상별 카메라 독립성 미확인 한계를 유지한다.

결과 폴더의 `report.json`에 라벨 근거·합의 기준·입력/소스 해시·환경·규칙 문턱·통계 구간, `scores.csv`·`per_video.csv`·`predictions.csv`·`paired_accuracy.csv`·`label_review.csv`에 전체 결과를 기록한다. 두 사람 중 한 명만 답했거나 제외한 행, 불일치 행은 합의 성적에 넣지 않는다. 평가 후 모델/문턱을 바꾸면 같은 시험을 새 독립 검증이라고 부르지 않는다.

## 누적 이진분류·2등급 비교 — 2026-11-04

사용자 지시로 **성능 개선이 검증될 때까지 최종 모델 결정은 보류**한다. 기존 G와 .623 두 단계는 비교 기준이며 최종 확정 모델로 취급하지 않는다. 딥러닝을 사용하지 않는다.

`nunchi_ml.glare.ordinal.CumulativeOrdinalClassifier`는 A=`y>=1`, B=`y>=2`를 각각 전체 학습 행으로 학습한다. `q1<q2`인 행은 두 값을 평균으로 풀링하는 최소제곱 등위회귀 투영을 적용한다. `[1-q1, q1-q2, q2]`의 argmax가 최종 등급이다. 원본 누적확률은 별도로 저장해 위반율과 각 이진분류기의 성능을 진단한다. 단일 클래스만 남은 학습 폴드는 상수 예측기로 처리하고 피처 이름·순서와 finite 입력을 검사한다. 기존 스케일링 Pipeline도 각 학습 폴드 안에서 fit한다.

```powershell
# 추가 알고리즘은 실험용 optional dependency, 운영 패키지 설치에 강제하지 않음
pip install -e ".[ordinal]"

# 기본 비교: 기존 320장/24촬영그룹, 같은 5겹×30회 + 카메라 제외 + 재사용 AI Hub
python notebooks/experiments/sun_visible_probe/ordinal_compare.py --out data/ordinal_20261104/primary30 --weighting balanced

# 추가 픽셀 특징/불균형 보정 비교. 별도 새 out 필요. 반복 수가 다르면 같은 seed끼리만 비교
python notebooks/experiments/sun_visible_probe/ordinal_compare.py --out data/ordinal_20261104/ablation5 --features g_rel g_rel_extra --weighting balanced none --repeats 5 --only-cv

# Optuna: 외부 fold의 학습 행만 안쪽 GroupKFold 3개로 나눠 A/B/다중분류 각각 최대2 trials
python notebooks/experiments/sun_visible_probe/ordinal_compare.py --out data/ordinal_20261104/nested_lgbm --families lightgbm --weighting balanced --trials 2 --workers 1

# 기존의 '심함은 G만' 입력을 유지한 진짜 누적분류 추가 대조. 전체 학습 행에서 B를 학습
python notebooks/experiments/sun_visible_probe/ordinal_severity_control.py --manifest data/ordinal_20261104/primary30/feature_manifest.csv --out data/ordinal_20261104/severity_control
```

- `.623`은 Accuracy가 아니라 기존 두 단계의 **320장 촬영 GroupKFold 5×30, 3등급 Macro F1 평균**이다. 기존 .5/.5 판정과 argmax 대조를 함께 기록한다. 새 다중분류/누적분류는 모두 argmax로 공정 비교한다.
- 이번30회 결과(Macro F1): 기존 두단계 .6229, CatBoost 다중 .4208→누적 .4421, XGBoost .4279→.4319, LightGBM .4477→.4647. 동일가중 누적 앙상블 .4558, 내부 Optuna2 trials LightGBM 누적 .4660, 심함 입력 G-only 로지스틱 누적 대조 최고 .5999. **.623 개선을 확인하지 못했으므로 최종 결정 보류.** 상세 등급별 지표·혼동행렬은 로컬 `data/ordinal_20261104/report.html`과 `combined_summary.csv`에 있다.
- 원본 스냅샷·사진·previous 날씨 결합을 유지하고 ID 일대일 매칭, 실제 사진 바이트 및 상대12 캐시의 재계산 일치를 확인한다. 평가용 카메라/촬영 ID를 피처에 넣지 않는다.
- `ordinal_image_features.py`는 평균·표준편차·P90/95/99·밝은 면적·배경 대비·중심·국소 대비·8구간 히스토그램을 사진별로만 계산한다. **정규화 픽셀 통계이며 물리적 휘도/조도나 센서 측정값이 아니다.** 배포 `features.py`를 바꾸지 않는다.
- `scores.csv`: Macro/Weighted F1, Accuracy, 등급별 P/R/F1, QWK, MAE, 0→2/2→0 건수, 혼동행렬. `threshold_scores.csv`: A/B 성능·Brier/log loss·단조성 위반. `predictions.csv`: 원본 q1/q2와 보정 후 클래스 확률. `splits.csv`·`parameters.json`·`method.json`: 분할·내부 튜닝 파라미터·입력/소스 해시·환경·완료 상태.
- `binary_scores.csv`는 0 vs 1·2 직접 분류(A, 보정 전 q1≥.5)와 3등급 결과를 합친 판정을 **같은 이진 과제**에서 비교한다. 2등급 F1을 3등급 .623과 직접 비교하지 않는다. 앙상블은 세 모델의 보정 후 클래스 확률을 같은 가중치로 평균하며 검증 라벨로 가중치를 고르지 않는다.
- 기본 파라미터는 기존 LightGBM 200 trees/learning rate .05/leaves7/min child5를 재사용한다. 기존 최적값이 없는 CatBoost/XGBoost는 200 trees/depth3/.05의 작은 탐색 기준이다. `--trials`는 0(고정) 또는 2~8로 제한한다. 각 이진 모델의 balanced 가중치는 해당 학습 표에서 따로 계산한다.
- 확률 보정(calibration), PCA, 자동 피처 선택, 외부 검증 라벨을 사용한 임계값 최적화는 이번 기본 비교에서 수행하지 않는다. 겹치는 30회 CV는 독립 신뢰구간이 아니다. AI Hub는 재사용 개발/스트레스 시험이고, 성진50장에는 G 시각/위치 입력이 없어 이 혼합 입력 모델을 평가하거나 튜닝하지 않는다. 4등급은 새로운 정의·사람 판정 없이 만들어내지 않는다.

## 후속: 촬영 그룹 내부 확률 보정

`CumulativeOrdinalClassifier(..., calibration="sigmoid").fit(X, y, groups=capture_groups)`로 선택적으로 사용한다. `CalibratedClassifierCV(ensemble=False)`가 각 바깥 학습 표 안의 GroupKFold3 OOF 예측만으로 보정하고, 기본 모델은 바깥 학습 표 전체로 다시 fit한다. `FoldWeightedClassifier`가 balanced 가중치를 각 **내부 학습 fold**에서 다시 계산하므로 내부 검증 라벨이 가중치에 유입되지 않는다. 보정 후에도 q1≥q2 투영·argmax를 적용한다. 기본값은 보정 없음이며 기존 예측을 유지한다. 그룹이 없거나2개 미만이면 보정은 거부한다. 근거: [scikit-learn 확률 보정 문서](https://scikit-learn.org/stable/modules/calibration.html).

```powershell
# 원래 학습 조건: 같은320장 그룹 CV + 카메라 제외 + 재사용 AI Hub
python notebooks/experiments/sun_visible_probe/ordinal_calibration_probe.py --out data/ordinal_20261104/calibration30

# 별도 데이터 추가 조건: AI Hub304장을 학습에 포함, 검증320장·분할은 그대로
# 이 조건의 AI Hub는 학습 데이터이며 AI Hub 시험 점수를 계산하지 않음
python notebooks/experiments/sun_visible_probe/ordinal_calibration_probe.py --out data/ordinal_20261104/calibration_augmented30 --augment-aihub
```

LightGBM·로지스틱·G-only 심함 입력 대조에서 raw/학습prior복원/group sigmoid를 비교한다. prior복원은 균형 가중치를 학습 클래스 비율로 대수적으로 되돌리는 대조이며 보정 성공을 보장하지 않는다. 별도 기존 두단계 대조에서는 내부 그룹 OOF만으로49개 임계값 조합을 비교한다. 누적분류의 argmax와 이 조건부 두단계 임계값 판정을 구분해서 기록한다. 결과는 `report.html`, `summary.csv`, `probability_summary.csv`, `binary_summary.csv`, `scores.csv`(등급별 지표·혼동행렬), `thresholds.csv`, `calibration_audit.csv`(내부 그룹/희소 클래스), `predictions.csv`, `splits.csv`, `method.json`, 실행 당시 바이트를 보존한 `source_snapshot/`에 있다. 중단·실패 상태도 기록한다.

30회 결과: 원래 학습 조건의 LightGBM 누적 Macro F1 .4647→group sigmoid .4562, 로지스틱 .5147→.4817, G-only 심함 대조 .5968→.4763. sigmoid 후 심함재현율은 모두0이었다. 기존 두단계 임계값 내부선택도 .6229→.4984로 악화했다. AI Hub304장 추가학습 조건은 같은 검증320장·분할에서 기존 두단계 .4133, G-only 심함 대조 .4965로 개선을 확인하지 못했다. 상세 비교는 로컬 `calibration_comparison.html`. 최종 모델 결정 보류를 유지한다.

확률 오차(Brier/log loss), 심함 재현율, Macro F1을 함께 확인한다. 보정이 F1 향상을 보장하지 않으며 데이터 추가 효과를 누적분류 알고리즘 효과로 해석하지 않는다. **현재 사용자 기준은 성진50장 단일 판정이며 추가 판정자를 실험 진행 조건으로 두지 않는다.** 다인 합의 정답/새 독립시험으로 해석하지 않는 한계와, 다른 촬영의 심함 표본 편중은 유지한다. 운영 모델·원본 라벨·분할을 바꾸거나 AI Hub 학습 성적을 외부 검증으로 보고하지 않는다.

## 후속: 사진상 전방 정보손실 특징 대조

2026-11-04 사용자 확정: 모델①은 통합 지도/경고, ②는 사진상 태양광에 따른 전방 정보 식별 저하, ③은 밝기 전환 순응 부담·예상 동공 변화 설명을 맡는다. 기존 사고 분석은 보존하며 실제 사고 확률 산출은 서비스 필수 조건에서 제외한다. 역할 정본과 검증 범위는 저장소 루트 README 및 Notion 모델링 계획서의 「현재 역할과 통합 원칙」을 따른다.

`road_visibility_features.py`는 고정 사다리꼴 관심 영역에서 국소 밝기 변동, 미세/넓은 영역 변동 비율, 작은 타일의 대비 분포, 윤곽 밀도와 선분 연속성 후보, 밝은/어두운 포화 비율, 밝은 번짐 중첩, 밝기 엔트로피 등12개를 추출한다. Canny는 영역 마스킹 전에 계산해 관심 영역 경계가 새 윤곽으로 생기지 않게 한다. Hough 선분은 도로 표시 후보이며 실제 차선 검출/식별성 정답이 아니다. 카메라 자세·도로 형태·노출·날씨·화질도 영향을 준다. 보정되지 않은 픽셀을 휘도/조도로 취급하지 않고 실제 운전자 시야·순응시간을 계산하지 않는다.

근거는 [Davoudian 등(2014)](https://doi.org/10.1177/1477153513510168)의 대비 손실, [Spieringhs 등(2025)](https://doi.org/10.1080/15502724.2024.2337385)의 도로 표시 식별, [Moumene·Benkedadra(2021)](https://doi.org/10.13052/jmm1550-4646.17414)의 정면 태양에서 카메라 전경 정보 손실, [Certad 등(2024)](https://arxiv.org/abs/2410.13320)의 표시 대비와 ADAS 관계다. 이 연구의 물리적 문턱·운전자 결과를 JPEG 등급 기준으로 이식하지 않는다. 신경망·이미지 복원·사전학습 임베딩을 사용하지 않는다.

```powershell
python notebooks/experiments/sun_visible_probe/road_visibility_probe.py --out data/road_visibility_20261104/run30 --repeats 30 --bootstrap 2000
```

- **기존320장/24촬영:** 같은 GroupKFold5×30(시드0..29)에서 legacy, 새12개를 여부 단계/조건부심함 단계/두 단계 모두에 추가, 이전 도로4개를 심함 단계에 넣는 대조, 누적 로지스틱 상대12/+새12개를 비교한다. 기존 .6229는 legacy의 .5/.5 판정이며 누적모델은 argmax다. 추가 특징 효과는 같은 구조·같은 판정끼리 비교한다.
- **카메라 제외·재사용 AI Hub304장:** 원래320장 학습에서 별도 스트레스 시험. AI Hub는 새 독립시험이 아니다.
- **사진만 보는624장→성진50장:** 기존 balanced LightGBM 화면12/+새12, balanced 로지스틱 상대12/+새12 및 새12만 비교한다. LightGBM은 argmax, 로지스틱은 기존 .5/.5 판정이다. 사람50장에는 G 입력이 없으므로 G를 채우지 않는다. 모두 고정 파라미터이며 사람50장 라벨을 이용한 튜닝/임계값 선택은 없다. 영상14개 단위 paired bootstrap은 추가 전후 Macro F1 차이를 기록한다. 이미 개발에 사용한 단일 판정50장이므로 새 독립시험/다인 합의가 아니다.
- 스케일링·균형 가중치는 학습 fold에서만 계산한다. 새 이미지 특징은 학습되지 않는 고정 함수이며 이미지 해시·특징 소스/버전 캐시를 검증한다. 기존 라벨·분할을 유지하고 동일 이미지가 학습/사람 평가에 중복되면 거부한다.
- 로컬 산출물: `summary.csv`, `scores.csv`(등급별Precision/Recall/F1·혼동행렬·QWK·MAE·0↔2오류), `cv_paired.csv`, `human_paired_bootstrap.csv`, `predictions.csv`, `splits.csv`, 특징표, `method.json`, `source_snapshot/`, `report.html`. 사진·정답·결과CSV는 공개하지 않는다. CV30회는 서로 겹치므로 독립 신뢰구간으로 해석하지 않는다. 최종 모델 선정은 개선 검증까지 보류한다.

### 실행 결과 · 2026-11-04

원본320장 촬영 GroupKFold5×30의 3등급 Macro F1 평균이다. `summary.csv`의 SD는 ddof=0이며 겹치는 반복 분할의 변동이다.

| 같은 정답·같은 분할 조건 | Macro F1 | 심함 재현율 | 중간등급 F1 |
|---|---:|---:|---:|
| 기존 두단계 | .6229 | .6562 | .6540 |
| 새12개를 여부 단계에 추가 | .6131 | .6877 | .6312 |
| 새12개를 심함 단계에 추가 | .4965 | .0785 | .6673 |
| 새12개를 두 단계에 추가 | .4849 | .0804 | .6489 |
| 이전 도로4개를 심함 단계에 추가 | .5778 | .4562 | .6567 |
| 상대12 누적 로지스틱 | .5147 | .1425 | .6666 |
| 상대12+새12 누적 로지스틱 | .4596 | .0023 | .6717 |

기존 .6229를 넘지 못했다. 여부 단계 추가는 30회 중8회만 좋아졌고 평균 차이 −.0098이다. 심함을 정상으로 놓치는 오류는 평균3.90→.80/320으로 줄었지만 정상→심함은16.70→17.93으로 늘었다. 심함 단계에 새 특징을 넣으면 심함을 주로 중간등급으로 내리는 실패가 발생했다. 누적모델 확률 역전은 상대12의8.28%→추가 후7.64%이며 모두 투영 보정했다.

재사용 AI Hub304장에서는 기존 .3215→두단계 모두 추가 .7142, 누적 상대12 .3066→+새12 .6683으로 개선했다. 두단계 모두 추가의 심함재현율은 .7143→.5476으로 낮아지고 중간등급 F1은0→.7927로 올랐다. 따라서 평균 F1 증가와 심함 누락의 교환이 있다. 기존320장CV에서는 반대 방향이며, 반복사용한 AI Hub 결과를 독립 일반화 성공이나 .6229 대비 개선으로 보고하지 않는다. 카메라제외 시험도 일관되지 않았다.

아래는 별도의 **사진만 보는624장 학습→성진50장 개발 평가**다. 320장 혼합 입력CV와 직접 비교하지 않는다.

| 사진 모델 | Macro F1 | 심함 재현율 | 중간등급 F1 | 눈부심 여부 정확도 |
|---|---:|---:|---:|---:|
| 화면12 LightGBM | .3179 | .20 | .0769 | .54 |
| 화면12+새12 LightGBM | .2861 | .10 | 0 | .56 |
| 상대12 로지스틱 | .2125 | 0 | .0714 | .54 |
| 상대12+새12 로지스틱 | .4508 | .50 | 0 | .64 |
| 새12만 로지스틱 | .3741 | .50 | 0 | .54 |
| 항상 정상 | .2602 | 0 | 0 | .64 |

상대 로지스틱은 +.2383이지만 영상14개 군집 paired bootstrap95% 범위는 [−.0427,+.3752]로0을 포함한다. 중간8장 전부 정상으로 놓쳤고, 심함10장 중5장만 심함으로 맞췄다. 눈부심 여부 정확도 .64는 항상 정상과 같다. 아래 혼동행렬의 행은 정답0/1/2, 열은 예측0/1/2다.

```text
상대12 (기존)       상대12+새12
15 10  7           24  7  1
 5  1  2            8  0  0
 1  9  0            2  3  5
```

사람 판정의 심함 일부를 잡는 후보 근거는 생겼으나, 안정적인3등급 개선·독립 일반화·실제 운전자 시야 저하 검증은 아니다. **최종선정/배포 교체 보류**를 유지한다. 기존320장·사람50장의 기준선을 모두 재현했고, 입력/라벨 해시 유지·확률 합계/비음수·30회 OOF/촬영 그룹 분리를 확인했다. 전체120개 테스트가 통과했다. 다음 우선순위는 중간등급8장의 오분류 원인 분석과 AI 보조 판정 후보의 블라인드 일치도 검증이며, 사람 라벨을 자동 수정하지 않는다.

### AI 보조 라벨링 후보 검증안 (새 판정 미실행)

강사 제안에 따라 Gemini 등 다른 이미지 이해 모델을 추가 라벨러 후보로 검토한다([공식 이미지 이해 문서](https://ai.google.dev/gemini-api/docs/image-understanding)). 기존 Claude 임시 라벨과 성진 판정은 공통48장 중34장 일치(70.8%, 카파 .473)했으며 AI의 우월성을 입증한 결과가 아니다. AI 라벨은 보조/학습 후보로 버전 분리하고 사람 기준 성적과 AI 라벨 재현 성적을 구분한다. 같은 AI를 여러 번 호출한 결과는 독립 판정자 수로 세지 않는다.

후보에게 사람 답안·기존 AI 답안·ML 예측·촬영 메타데이터를 숨기고 동일 원본 사진/고정 기준으로 아래 항목을 기록하게 한다. 사람이 판정한 것으로 선언하거나 실제 운전자의 생리 상태를 추정하게 하지 않는다. 불확실하거나 원인이 구분되지 않으면 판단 보류를 허용한다. 기존 기준을 바꿀 때는 새 라벨 버전으로 분리하고 .6229와 직접 비교하지 않는다.

```text
주행 사진에서 관찰 가능한 증거만 판정한다. 사진 밖 정보나 원래 있었을
객체를 추측하지 않는다. 기존 정답·다른 모델 답안은 제공되지 않는다.
1. 태양 원반 또는 광원에서 시작되는 번짐의 근거: 있음/없음/불확실.
2. 차선·앞차·표지판 각각: 식별 가능/식별 저하/대상 없음/판단 불가.
3. 저하 원인: 태양광/안개·비/흔들림·저해상도/가림/기타/불확실.
4. 기존 등급 기준에 따라 0/1/2 또는 판단 보류와 짧은 관찰 근거를 기록.
   0: 태양광 눈부심 시각 근거 없음.
   1: 태양·번짐 근거가 있으나 주요 전방 정보는 식별 가능.
   2: 태양광으로 주요 전방 정보 식별이 현저히 저하됨.
백색 하늘·객체 부재·단순 화질 저하만으로 태양광의 심함을 확정하지 않는다.
실제 휘도, 운전자 동공 크기, 적응시간, 사고확률은 이 사진으로 추정하지 않는다.
```

실행 시 모델의 정확한 ID/버전·프롬프트/설정·이미지 해시·판정 일시·보류 여부·근거를 보존한다. 후보 평가에는 사람과의 일치율·가중 카파·등급별 성적·심한 불일치·판정 커버리지를 함께 기록한다. 개발에 사용한50장의 일치도가 높아도 일반화/실제 운전자 정답을 입증하지 못한다. 이 문서 수정으로 추가 AI 판정을 실행하거나 기존 라벨을 교체하지 않았다.

## 한계 (알고 쓴다)
- 사람이 판정한 해 가시성 · 가림 정답이 없는 **대리 특징**이다. 새 카메라에서 해가 보일 때의 재현율은 증명되지 않았다 (96장 블라인드 검수 판정 전).
- **도로 광각 카메라에서는 나빠진다** (알림정확도 .86 → .84, 헛알림 22% → 39%). 포화된 하늘 가장자리를 해로 오인하는 사례가 있다 (PR #16 시각 검토).
- 심한 눈부심 정밀도는 어떤 방법도 개선하지 못했다 (재사용 AI Hub .09~.15, G 도 .146).
- AI Hub 304장은 개발 과정에서 반복 사용한 세트이고, 특징은 AI Hub 실패 원인을 본 뒤 설계했다 → **독립 외부 검증이 아니다.** 한 주행(서울2 drive_1189)이 75장.
- `rel_features.py` 와 `nunchi_ml/glare/visibility.py`(PR #16)는 일부(6개)가 겹친다. 이 폴더는 비교용 초안이며 패키지로 옮길지는 결정하지 않았다.
