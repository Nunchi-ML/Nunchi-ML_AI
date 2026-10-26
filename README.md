# Nunchi-ML_AI

데이터 정제 · EDA · 모델 실험 · 최종 파이프라인 저장소입니다.
담당: 데이터 엔지니어 A · B, AI 엔지니어 A · B

## 폴더

```
data/        원본 · 정제 데이터 (git에 올리지 않음, 각자 로컬 · 드라이브)
notebooks/   eda/ · experiments/ 노트북
src/         여러 노트북이 같이 쓰는 코드 (불러오기 · 정제 · 분할 · 파이프라인)
models/      저장한 파이프라인 파일 (git에 올리지 않음 → Release로 전달)
```

## 기준 (노션이 정본)

- 검증 기준: 노션 「분석계획서」 — 평가 지표 · fold 수 · 시드
- 분할 파일 `folds_v1`: 노션 「데이터 파이프라인」 4단계에서 생성, **모든 실험은 이 파일만 쓴다**
- 데이터셋 버전(v1, v2 …)이 다르면 점수를 비교하지 않는다
- 실험은 노션 「실험 기록」에 한 줄씩 (번호 EXP-)

## 모델 넘기기 (AI → BE)

1. 전처리 + 모델을 sklearn `Pipeline` 하나로 묶어 `joblib`으로 저장한다
2. 파이프라인 안에는 sklearn · LightGBM · XGBoost · CatBoost 기본 부품만 쓴다 (직접 만든 클래스가 들어가면 BE에서 파일이 안 열린다)
3. 이 저장소 Release에 파일을 붙인다 (태그 예: `model-v1`), 노션 「API 명세」 모델 교체 기록에 실험 번호를 적는다
4. 저장할 때 쓴 scikit-learn 버전을 Release 설명에 적고, BE `requirements.txt`도 같은 버전으로 맞춘다

## 실행

```bash
pip install -r requirements.txt
```

## 작업 규칙

- `main`에 바로 올리지 않고 브랜치 → PR → 1명 이상 승인 → squash 병합
- 브랜치: `feat/` 기능 · 실험, `fix/` 버그, `docs/` 문서, `refactor/` 정리, `data/` 데이터 처리 + 짧은 설명 (예: `feat/lgbm-baseline`)
- 커밋: `feat: 베이스라인 모델 추가`처럼 접두어 + 한 줄
- 착수 전 슬랙 `[작업 시작]`, 병합 후 `[작업 정리]` + 노션 카드에 PR 링크
- 리뷰 코멘트: P1 꼭 고침 · P2 고치면 좋음 · P3 의견
