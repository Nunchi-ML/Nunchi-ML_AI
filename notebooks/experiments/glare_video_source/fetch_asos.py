"""기상청 ASOS 대구(143) 시간자료 받기 → 사진 촬영 시각에 구름양 · 일조 · 일사 붙이기.

공공데이터포털 '기상청_지상(종관, ASOS) 시간자료 조회서비스' 키를 AI/.env 에 KMA_SERVICE_KEY=... (Decoding 키, git 제외)
받는 칸:
    dc10Tca  전운량 (0 맑음 ~ 10 흐림)
    ss       일조 (그 정시까지 한 시간 동안 해가 비친 시간, 0~1)
    icsr     일사 (MJ/m², 그 한 시간)
    vs       시정 (10 m 단위 — 안개 · 미세먼지로 빛이 퍼지는 정도)
    rn       강수 (mm, 빈칸 = 0 — 젖은 도로 반사)

    python notebooks/experiments/glare_video_source/fetch_asos.py
결과: data/asos/daegu_143.csv (날짜별로 받아 쌓음 — 이미 받은 날은 다시 안 물음). 서울은 attach(d, stn=108) → seoul_108.csv
      사진에 붙이기는 attach() — 촬영 시각의 가장 가까운 정시 값
"""
import time
from pathlib import Path

import pandas as pd
import requests

import find_sun_frames as F

API = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
CACHE = F.ROOT / "data" / "asos" / "daegu_143.csv"
STATIONS = {143: "daegu", 108: "seoul"}


def cache(stn):
    return F.ROOT / "data" / "asos" / f"{STATIONS[stn]}_{stn}.csv"
COLS = {"dc10Tca": "cloud", "ss": "sunshine", "icsr": "radiation", "vs": "visibility", "rn": "rain"}


def key():
    env = F.ROOT / ".env"
    for line in env.read_text(encoding="utf-8").splitlines() if env.exists() else []:
        if line.startswith("KMA_SERVICE_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("KMA_SERVICE_KEY 가 없어요 — AI/.env 에 한 줄")


def fetch_day(day, k, stn=143):
    p = {"serviceKey": k, "pageNo": 1, "numOfRows": 30, "dataType": "JSON", "dataCd": "ASOS", "dateCd": "HR",
         "startDt": day, "startHh": "00", "endDt": day, "endHh": "23", "stnIds": str(stn)}
    for attempt in range(4):
        try:
            r = requests.get(API, params=p, timeout=60); r.raise_for_status()
            items = r.json()["response"]["body"]["items"]["item"]
            return pd.DataFrame(items)[["tm"] + list(COLS)].rename(columns=COLS)
        except (requests.RequestException, KeyError, ValueError) as e:
            print("  재시도:", day, e); time.sleep(3 * (attempt + 1))
    raise SystemExit(f"ASOS 응답 없음: {day}")


def ensure(days, stn=143):
    CACHE = cache(stn)
    have = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["tm", *COLS.values()])
    if not set(COLS.values()) <= set(have.columns):                  # 받는 칸이 늘면 처음부터 다시
        have = pd.DataFrame(columns=["tm", *COLS.values()])
    got = set(have.tm.str[:10].str.replace("-", ""))
    todo = sorted(set(days) - got)
    k = key()
    for n, day in enumerate(todo, 1):
        have = pd.concat([have, fetch_day(day, k, stn)], ignore_index=True)
        print(f"  ASOS {n}/{len(todo)} {day}", flush=True)
        time.sleep(0.2)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    have.drop_duplicates("tm").sort_values("tm").to_csv(CACHE, index=False, encoding="utf-8-sig")
    return have


def attach(d, time_col="captured_at_kst", stn=143):
    """d[time_col](한국시간 문자열) 에 가장 가까운 정시의 cloud · sunshine · radiation 을 붙임. 빈 값은 NaN."""
    t = pd.to_datetime(d[time_col])
    a = ensure(t.dt.strftime("%Y%m%d").unique(), stn)
    a = a.assign(tm=pd.to_datetime(a.tm)).set_index("tm")
    for c in COLS.values():
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a.loc[a.index.hour.isin(range(7, 19)), "sunshine"] = a.sunshine.fillna(0)   # 낮에 일조 빈칸 = 해 안 남
    a["rain"] = a.rain.fillna(0)                                                  # 강수 빈칸 = 비 안 옴
    hour = t.dt.round("h")
    return d.assign(**{c: hour.map(a[c]).to_numpy() for c in COLS.values()})


def main():
    days = set()
    for f in F.ROOT.glob("data/glare_mapillary*/candidates.csv"):
        days |= set(pd.to_datetime(pd.read_csv(f).captured_at_kst).dt.strftime("%Y%m%d"))
    a = ensure(days)
    print(f"대구 ASOS {a.tm.str[:10].nunique()}일 · {len(a)}시간 → {CACHE}")


if __name__ == "__main__":
    main()
