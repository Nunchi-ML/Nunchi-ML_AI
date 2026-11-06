import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'notebooks/experiments/sun_visible_probe'))
    return importlib.import_module('sd_weather_photo_probe')


def weather_fixture():
    frame=pd.DataFrame(dict(stable_id=['b','a'],filename_datetime=['2026-11-03 08:10:00']*2))
    weather=pd.DataFrame(dict(stable_id=['a','b'],cloud=[0.,10.],sunshine=[0.,.5],
        weather_observed_at=['2026-11-03 08:00:00']*2,weather_requested_hour=['2026-11-03 08:00:00']*2,
        weather_age_minutes=[10.,10.],weather_availability_verified=[False,False]))
    return frame,weather


def test_weather_joins_by_id_preserves_order_real_zeros_and_source(module):
    frame,weather=weather_fixture();before=weather.copy(deep=True)
    result=module.attach_weather(frame,weather)
    assert result.stable_id.tolist()==['b','a']
    assert result.cloud.tolist()==[10.,0.] and result.sunshine.tolist()==[.5,0.]
    assert not result.weather_availability_verified.any()
    pd.testing.assert_frame_equal(weather,before)


def test_weather_never_imputes_missing_or_ambiguous_observations(module):
    frame,weather=weather_fixture()
    for bad in [weather.iloc[:1],pd.concat([weather,weather.iloc[:1]]),weather.assign(sunshine=np.nan),weather.assign(cloud=11)]:
        with pytest.raises(ValueError):module.attach_weather(frame,bad)


def test_weather_checks_timestamps_even_if_serialized_age_claims_past(module):
    frame,weather=weather_fixture()
    for hour in ['2026-11-03 09:00:00','2026-11-03 07:00:00']:
        bad=weather.assign(weather_observed_at=hour,weather_requested_hour=hour)
        with pytest.raises(ValueError,match='preceding capture hour'):module.attach_weather(frame,bad)
    with pytest.raises(ValueError,match='nonfuture'):
        module.attach_weather(frame,weather.assign(weather_age_minutes=-1))


def test_binary_error_flags_do_not_count_severity_underestimate_as_missed_alert(module):
    assert module.error_flags([0,1,2,2,1],[2,0,1,0,1]).tolist()==['FP_GPT','FN_GPT','grade_disagreement','FN_GPT','agreement']


def test_review_selection_is_balanced_reproducible_and_avoids_shared_clips(module):
    records=[]
    for s,subset in enumerate(['initial300','content69']):
        for k,(label,pred,error) in enumerate([(0,1,'FP_GPT'),(0,2,'FP_GPT'),(1,0,'FN_GPT'),(2,0,'FN_GPT'),(0,0,'agreement'),(1,1,'agreement')]):
            records.append(dict(stable_id=f'{s}_{k}',subset=subset,clip_id=f'clip{s}_{k}',label=label,screen_lgb=pred,error=error,p0=.1 if pred else .9))
    frame=pd.DataFrame(records);result=module.select_cases(frame)
    assert len(result)==12 and result.clip_id.nunique()==12
    assert result.groupby(['subset','review_kind']).size().tolist()==[2,2,1,1]*2
    pd.testing.assert_frame_equal(result,module.select_cases(frame.sample(frac=1,random_state=1)))
    with pytest.raises(ValueError,match='Insufficient'):
        module.select_cases(frame.assign(clip_id='same_clip'))
