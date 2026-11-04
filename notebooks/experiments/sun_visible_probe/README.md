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

## 한계 (알고 쓴다)
- 사람이 판정한 해 가시성 · 가림 정답이 없는 **대리 특징**이다. 새 카메라에서 해가 보일 때의 재현율은 증명되지 않았다 (96장 블라인드 검수 판정 전).
- **도로 광각 카메라에서는 나빠진다** (알림정확도 .86 → .84, 헛알림 22% → 39%). 포화된 하늘 가장자리를 해로 오인하는 사례가 있다 (PR #16 시각 검토).
- 심한 눈부심 정밀도는 어떤 방법도 개선하지 못했다 (재사용 AI Hub .09~.15, G 도 .146).
- AI Hub 304장은 개발 과정에서 반복 사용한 세트이고, 특징은 AI Hub 실패 원인을 본 뒤 설계했다 → **독립 외부 검증이 아니다.** 한 주행(서울2 drive_1189)이 75장.
- `rel_features.py` 와 `nunchi_ml/glare/visibility.py`(PR #16)는 일부(6개)가 겹친다. 이 폴더는 비교용 초안이며 패키지로 옮길지는 결정하지 않았다.
