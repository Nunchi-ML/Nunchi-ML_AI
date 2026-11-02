# 해가 정면에 낮게 있던 도로 사진 찾기 (모델 ② 데이터 · G 확인용)

GLARE(미국 블랙박스 2,157장)에는 **찍은 시각 · 위치가 없어서** 해 높이와 방향을 알 수 없어요.
촬영 시각 · GPS · 카메라 방향이 함께 있는 사진을 찾아, 계산한 해 위치로 "해가 정면에 낮게 있던" 장면을 고릅니다.

| 쓰임 | 내용 |
|---|---|
| 모델 ② | 한국 도로 사진으로 학습 · 검증 사진 늘리기 (2인 라벨) |
| 모델 ③ G 확인 | 해 높이 · 방향 차이 Δ 와 실제 눈부심 단계의 관계 → 재혁님 공식 G 가 맞는지 |

## 데이터 후보 (2026-11-02 확인)

| 데이터 | 시각 · GPS · 방향 | 받기 | 메모 |
|---|---|---|---|
| **Mapillary** (채택) | ✅ 사진마다 | 무료 개발자 토큰 | 대구 도로 포함. 사진 CC BY-SA 4.0 (출처 표기) |
| BDD100K | ✅ (info JSON) | ❌ 공식 다운로드 서버(dl.cv.ethz.ch)가 없어짐 | 거울 사이트에도 GPS 파일 없음 |
| comma2k19 | ✅ | Hugging Face (MIT) | 미국 고속도로만, 10GB 단위 |
| AI Hub 자율주행 영상 | (확인 필요) | 신청 · 승인 | 한국 도로 |

## 실행

1. https://www.mapillary.com/dashboard/developers 에서 앱 만들고 **Client Token** 복사
2. AI 폴더 `.env` 에 한 줄 (git 에 안 올라감): `MAPILLARY_TOKEN=...`
3. ```bash
   pip install -e .
   python notebooks/experiments/glare_video_source/find_sun_frames.py --route ../FE/src/data/dalgu.json
   ```
4. 결과 `data/glare_mapillary/candidates.csv` (git 제외)

| 열 | 뜻 |
|---|---|
| `captured_at_kst` · `lat` · `lon` · `heading` | 촬영 시각(한국시간) · 위치 · 카메라 방향(북=0) |
| `sun_alt` · `sun_az` | 그 시각 · 위치의 해 높이 · 방위 (`nunchi_ml.sun`, pvlib) |
| `gap` | 해 방위 − 카메라 방향 차이 Δ (0 = 정면) |
| `front_low` | 정면 저각도 (해 높이 0~25° · Δ ±25°) — 모델 ① · ③ 과 같은 기준 |
| `sequence` | 연속 촬영 묶음 → 검증 때 **묶음 단위로 나누기** (같은 영상 사진이 학습 · 시험에 섞이지 않게) |

## 다음
- 정면 저각도 · 비교군(같은 묶음의 다른 방향 · 높은 해) 사진을 골라 내려받기 → 2인 라벨
- 라벨 기준은 노션 분석계획서 「모델 ② 라벨 기준표」 (불일치 67장 재검토 후 고친 판)
