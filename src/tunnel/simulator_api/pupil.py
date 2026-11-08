"""동공 모델 (sunsight_pupil/02_터널순응_위험도표.ipynb와 같은 식). 터널 입구(암순응)만.
정상 상태 동공 크기 = Watson & Yellott(2012), 시야 3,000deg².
경계부 끝(80m ÷ 속도) 동공 = D전 + 열림비율 × (D후 − D전), 열림비율 = 01 노트북 PsPM 2단 공식 (회색→0, 전체 평균) + τ2 가정.
동공 칸은 등급을 바꾸지 않는 설명용이다. 잔여조도비_가정 = 휘도비 ÷ 동공 면적비 (산술값, 시야 손실·사고 확률 아님)."""
from pathlib import Path
import numpy as np, pandas as pd

AREA = 3000
TAU2 = {'빠름': 10, '기본': 30, '느림': 60}
_par = pd.read_csv(Path(__file__).with_name('동공시간_공식숫자.csv'), encoding='utf-8-sig')
DARK = _par[(_par.대상 == '전체 평균') & (_par.전환 == '회색→0')].iloc[0]


def wy(L, age, area=AREA):
    """Watson & Yellott 2012 동공 지름(mm). L = 휘도 cd/m²."""
    x = (np.asarray(L, float) * area / 846) ** 0.41
    D = 7.75 - 5.75 * x / (x + 2)
    return D + (age - 28.58) * (0.02132 - 0.009562 * D)


def frac_dark(t, tau2):
    r = DARK; x = max(t - r.d, 0); A2 = r.b * tau2
    return (r.A1 * (1 - np.exp(-x / r.tau1)) + A2 * (1 - np.exp(-x / tau2))) / (r.A1 + A2)


def pupil_entrance(L_out, L_in, age=40, tau2='기본', speed_kmh=100, zone_m=80):
    t = zone_m / (speed_kmh / 3.6)
    D_pre, D_post = float(wy(L_out, age)), float(wy(L_in, age))
    f = float(frac_dark(t, TAU2[tau2]))
    D_zone = D_pre + f * (D_post - D_pre)
    ratio = L_out / L_in
    a_full = max(D_pre, D_post) ** 2 / min(D_pre, D_post) ** 2
    a_zone = max(D_pre, D_zone) ** 2 / min(D_pre, D_zone) ** 2
    return {'age': age, 'tau2_assumption': tau2, 'boundary_time_s': round(t, 2),
            'pupil_outside_mm': round(D_pre, 3), 'pupil_inside_full_mm': round(D_post, 3), 'pupil_boundary_end_mm': round(D_zone, 3),
            'open_fraction_1s': round(float(frac_dark(1, TAU2[tau2])), 3), 'open_fraction_2s': round(float(frac_dark(2, TAU2[tau2])), 3),
            'open_fraction_boundary_end': round(f, 3),
            'pupil_area_ratio_full': round(a_full, 3), 'pupil_area_ratio_boundary_end': round(a_zone, 3),
            'residual_ratio_full': round(ratio / a_full, 2), 'residual_ratio_boundary_end': round(ratio / a_zone, 2),
            'note': '동공은 설명용(등급을 바꾸지 않음). residual = 휘도비 ÷ 동공 면적비, 망막이 감당할 몫의 산술값'}
