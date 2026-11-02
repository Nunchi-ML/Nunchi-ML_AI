"""공통 해 위치 기준 — 상식으로 확인할 수 있는 값들."""
import numpy as np

from nunchi_ml.sun import heading_gap, is_front_low, sun_position

DAEGU = (35.857, 128.630)          # 대구은행 본점 부근


def test_morning_sun_is_low_in_the_east_and_noon_is_high_in_the_south():
    pos = sun_position(["2026-11-02 07:30", "2026-11-02 12:30"], DAEGU[0], DAEGU[1])
    morning, noon = pos.iloc[0], pos.iloc[1]
    assert 0 < morning.altitude < 15 and 90 < morning.azimuth < 130      # 11월 아침: 동남동 낮게
    assert 35 < noon.altitude < 45 and 165 < noon.azimuth < 200          # 정오 무렵: 남쪽 약 40°


def test_night_sun_is_below_horizon():
    assert sun_position(["2026-11-02 23:00"], *DAEGU).altitude.iloc[0] < 0


def test_heading_gap_wraps_around_north():
    assert np.allclose(heading_gap([350, 10, 90], [10, 350, 270]), [20, 20, 180])


def test_front_low_rule():
    assert list(is_front_low([10, 10, 30, -2], [5, 40, 5, 5])) == [True, False, False, False]
