# SunSight Model 1 — VDS 교통량 결합 및 Offset 보정 분석 계획

## 1. 배경 및 목적
- **기존 한계**: 기존 Model 1은 1km 구간당 '사고 건수($Y_i$)'를 예측하므로, 교통량이 많은 노선(경부선 등)의 통행량 효과가 순수한 역광 위험도와 혼재됨.
- **개선 목표**: 한국도로공사 VDS(차량검지기) 일평균 교통량(AADT)을 결합하여, **교통 노출량($E_i$) 대비 사고 발생률(Rate per VKT)**을 예측하는 Poisson Offset 모델링 구축.

$$\log(\lambda_i) = \log(E_i) + \beta_0 + \sum_{k} \beta_k X_{ik}$$

## 2. 사용 데이터 출처 및 연계 경로
1. **고속도로 구간 기준**: `data/expressway/segment_1km_dataset_v1.csv` (전국 61개 노선, 5,782개 구간)
2. **VDS 지점/존 마스터**: 공공데이터포털 / 한국도로공사 교통데이터 오픈마켓
   - 파일: 고속도로 지점별 교통 소통 통계 / VDS 마스터 (`route`, `start_mp`, `end_mp`, `vds_zone_id`)
3. **교통량(AADT)**: 2023~2025년 3개년 평균 또는 연도별 합산 통행량

## 3. 매핑 및 전처리 규칙
- **이정표 기반 구간 인터벌 조인**:
  - 각 1km 구간의 중심 이정 $\text{mid\_mp} = (\text{start\_mp} + \text{end\_mp}) / 2$
  - VDS 존 $[\text{start\_mp}, \text{end\_mp}]$ 범위에 포함되는 존의 AADT 평균 매핑
- **교통 노출량(VKT) 산출**:
  $$E_i = \text{AADT}_i \times \text{segment\_length}_i \times 365 \times 3$$
  $$\text{Offset} = \log(E_i)$$
- **결측 및 미설치 구간 처리**:
  - VDS 미설치 구간/신규 개통 구간: 동일 노선 인접 VDS 선형 보간 또는 노선 중앙값 대치

## 4. 모델 비교 및 평가 지표 (GroupKFold 5-Fold)
| 평가 지표 | 기존 Model 1 (건수 기준) | Offset Model (사고율 기준) | 의의 |
|---|---|---|---|
| Poisson Deviance | 0.1299 | TBD | 노출량 대비 설명력 변화 |
| 위험 구간 순위 변동 | 건수 중심 랭킹 | 노출 대비 사고율 랭킹 | 순수 도로 위험 구간 재발굴 |
| 상위 10% 포착률 | 16.2% | TBD | 위험 포착 효율 검증 |
| 피처 가중치/중요도 | 건수 설명 피처 | 사고율 설명 피처 | 기하학적 역광 순수 기여도 분리 |

## 5. 실행 파이프라인 모듈
- 매핑 로직: `src/nunchi_ml/accident/vds_offset.py`
- 실험 노트북/스크립트: `notebooks/experiments/accident_pilot/08_vds_offset_experiment.py` (준비 예정)
