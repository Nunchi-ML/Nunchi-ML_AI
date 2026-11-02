"""모델 ③ 터널 순응 부담 점수 A (진입 = 암순응, 진출 = 명순응).

※ 업데이트 중인 초안 — 계산식 · 상수는 아직 검토 전이라 바뀔 수 있다.

일사량(ASOS icsr, MJ/m², 1시간 누적) 하나로 바깥 밝기를 구하고,
국토부 「도로안전시설 설치 및 관리지침 — 조명시설 편」 6장의 터널 안 밝기와 비교한다.

    W/m²  = 일사량 × 277.8            (1 MJ ÷ 3,600초)
    E(lx) = W/m² × 126                (서울 일사 발광효율, 윤갑천 외 2011)
    L20   = 0.15 × E ÷ π              (바깥 휘도 cd/m², 반사율 0.15는 가정)
    경계부 = 280 × 조광 단계            (4,000 × 0.07, 표 6.4 · 6.5 / 조광 25·50·75·100%, 표 6.7)
    출구부 = 45                        (기본부 9 × 5, 표 6.2 + CIE 규칙)
    A     = log10(L20 ÷ 터널 안 휘도) ÷ 2, 0~1로 자름 (휘도비 100배 = 1)

사용:
    python -m src.tunnel.adaptation 3.0        # 일사량 3.0 MJ/m² 예시 출력
"""
import sys

import numpy as np

MJ_TO_W = 1_000_000 / 3600      # 277.8
EFFICACY = 126                  # lm/W
REFLECT = 0.15                  # 노면 · 주변 반사율 (가정)
L_THRESHOLD = 280               # 경계부 휘도, 100 km/h
L_EXIT = 45                     # 출구부 휘도

ENTRY_ZONES = ("진입전150", "경계100")
EXIT_ZONES = ("출구60", "진출후150")


def outside_luminance(icsr):
    """일사량(MJ/m²) → 바깥 휘도 L20(cd/m²)."""
    icsr = np.nan_to_num(np.asarray(icsr, float))
    return REFLECT * icsr * MJ_TO_W * EFFICACY / np.pi


def dimming_step(L20):
    """조광 4단계: L20 ÷ 4,000을 25% 단위로 올림, 25~100%."""
    return np.clip(np.ceil(np.asarray(L20, float) / 4000 * 4) / 4, 0.25, 1)


def score(R):
    """휘도비 R → 0~1 점수."""
    R = np.asarray(R, float)
    return np.clip(np.log10(np.where(R > 0, R, 1)) / 2, 0, 1)


def adaptation_scores(icsr, tunnel_zone=None):
    """(A_진입, A_진출). tunnel_zone을 주면 해당 구간 밖은 0."""
    L20 = outside_luminance(icsr)
    a_in = score(L20 / (L_THRESHOLD * dimming_step(L20)))
    a_out = score(L20 / L_EXIT)
    if tunnel_zone is not None:
        z = np.asarray(tunnel_zone, object)
        a_in = np.where(np.isin(z, ENTRY_ZONES), a_in, 0.0)
        a_out = np.where(np.isin(z, EXIT_ZONES), a_out, 0.0)
    return a_in, a_out


if __name__ == "__main__":
    for v in map(float, sys.argv[1:] or ["3.0", "1.0", "0.3", "0"]):
        L = outside_luminance(v)
        a_in, a_out = adaptation_scores(v)
        print(f"일사량 {v:.1f} → L20 {L:,.0f} cd/m², 조광 {dimming_step(L):.0%}, "
              f"A_진입 {a_in:.2f}, A_진출 {a_out:.2f}")
