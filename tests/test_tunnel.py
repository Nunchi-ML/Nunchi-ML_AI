"""모델 ③ 터널 순응 점수 A — 계산식이 바뀌어도 기본 성질은 지켜야 한다."""
import numpy as np

from nunchi_ml.tunnel.adaptation import adaptation_scores


def test_no_sun_means_no_burden():
    a_in, a_out = adaptation_scores(0.0)
    assert float(a_in) == 0 and float(a_out) == 0


def test_scores_stay_between_0_and_1():
    a_in, a_out = adaptation_scores(np.array([0.0, 0.5, 1.5, 3.0, 4.0]))
    for a in (a_in, a_out):
        assert np.all((a >= 0) & (a <= 1))


def test_brighter_sky_never_lowers_burden():
    icsr = np.linspace(0, 4, 41)
    a_in, a_out = adaptation_scores(icsr)
    assert np.all(np.diff(a_out) >= -1e-9)
