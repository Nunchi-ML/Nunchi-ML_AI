"""Offline weather-time replay on one frozen snapshot, holding geometry fixed.

The snapshot needs id, label, camera, group, captured_at_kst, sun_alt/u/v/in_fov,
cloud/sunshine and area for AI Hub rows. Raw data/results stay outside Git.
ASOS cache timestamps prove observation time only, not real-time availability.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nunchi_ml.weather import WEATHER_COLUMNS, align_hourly_weather

G_COLUMNS = ["sun_alt", "u", "v", "in_fov", "cloud", "sunshine"]
STATIONS = {143: "daegu", 108: "seoul", 133: "daejeon", 155: "changwon"}


def replay(snapshot, cache_dir, alignment):
    """Local CSVs only; use the same legacy missing-value handling as attach()."""
    # A CSV exported by a newer loader may already contain serialized metadata.
    # Recompute it instead of trusting old policy/publication flags or dtypes.
    result = snapshot.drop(columns=[c for c in snapshot if c.startswith("weather_")]).copy()
    if not set(snapshot.camera) <= {"도로 광각", "블랙박스", "서울 폰", "AI Hub"}:
        raise ValueError("Unknown camera/station mapping")
    station = pd.Series(143, index=snapshot.index)
    station[snapshot.camera.eq("서울 폰")] = 108
    ai = snapshot.camera.eq("AI Hub")
    if not set(snapshot.loc[ai, "area"]) <= {"seoul2", "daejeon", "changwon"}:
        raise ValueError("Unknown AI Hub station mapping")
    for area, stn in [("seoul2", 108), ("daejeon", 133), ("changwon", 155)]:
        station[ai & snapshot.area.eq(area)] = stn
    for stn in station.unique():
        weather = pd.read_csv(cache_dir / f"{STATIONS[stn]}_{stn}.csv")
        weather[list(WEATHER_COLUMNS)] = weather[list(WEATHER_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        day = pd.to_datetime(weather.tm).dt.hour.isin(range(7, 19))
        weather.loc[day, "sunshine"] = weather.sunshine.fillna(0)
        weather.rain = weather.rain.fillna(0)
        m = station.eq(stn)
        matched = align_hourly_weather(snapshot.loc[m, "captured_at_kst"], weather, alignment=alignment)
        # Add metadata columns with their proper dtype before assigning a subset.
        for col in matched:
            if col not in result:
                if pd.api.types.is_datetime64_any_dtype(matched[col]):
                    result[col] = pd.NaT
                elif col == "weather_availability_verified":
                    result[col] = False
                elif col == "weather_time_policy":
                    result[col] = alignment
                else:
                    result[col] = np.nan
            result.loc[m, col] = matched[col].to_numpy()
    return result


def predict(train, test):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    model.fit(train[G_COLUMNS], train.label)
    p = np.zeros((len(test), 3))
    p[:, model.classes_] = model.predict_proba(test[G_COLUMNS])
    return np.where(p[:, 1:].sum(1) >= .5, np.where(p[:, 2] >= p[:, 1], 2, 1), 0)


def metrics(y, pred):
    y = np.asarray(y)
    return {"macro_f1": f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0),
            "alert_accuracy": accuracy_score(y >= 1, pred >= 1),
            "severe_precision": precision_score(y, pred, labels=[2], average="macro", zero_division=0),
            "severe_recall": recall_score(y, pred, labels=[2], average="macro", zero_division=0),
            "false_alert_rate": float((pred[y == 0] >= 1).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parents[3] / "data/asos")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    snapshot = pd.read_csv(args.snapshot)
    variants = {mode: replay(snapshot, args.cache_dir, mode) for mode in ("nearest", "previous")}
    # Identical rows/splits for both policies; never silently drop a missing hour.
    valid = np.logical_and.reduce([d[G_COLUMNS].notna().all(axis=1) for d in variants.values()])
    if not valid.all():
        raise ValueError(f"Missing weather in {int((~valid).sum())} rows; resolve coverage before scoring")
    rows, audit, predictions = [], [], []
    for mode, d in variants.items():
        ah = d.camera.eq("AI Hub")
        train, test = d[~ah].reset_index(drop=True), d[ah]
        p = predict(train, test)
        rows.append({"alignment": mode, "test": "reused_AIHub", **metrics(test.label, p)})
        predictions.extend({"alignment": mode, "id": i, "label": int(y), "pred": int(q)}
                           for i, y, q in zip(test.id, test.label, p))
        for seed in range(args.repeats):
            p = np.zeros(len(train), dtype=int)
            for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(train, train.label, train.group):
                p[te] = predict(train.iloc[tr], train.iloc[te])
            rows.append({"alignment": mode, "test": "group_cv", "repeat": seed, **metrics(train.label, p)})
        audit.append({"alignment": mode, "rows": len(d), "future_observation_rows": int((d.weather_age_minutes < 0).sum()),
                      "publication_verified_rows": int(d.weather_availability_verified.sum())})
    args.out.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(args.out / "scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(args.out / "predictions.csv", index=False, encoding="utf-8-sig")
    for mode, d in variants.items():
        d[["id", *WEATHER_COLUMNS, *[c for c in d if c.startswith("weather_")]]].to_csv(
            args.out / f"alignment_{mode}.csv", index=False, encoding="utf-8-sig")
    info = {"snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(), "sklearn": sklearn.__version__,
            "repeats": args.repeats, "decision_thresholds": [.5, .5], "audit": audit,
            "cache_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.cache_dir.glob("*.csv")},
            "limit": "Reused external set. Observation-time replay only; publication times unknown. Not live-service validation."}
    (args.out / "method.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(scores.groupby(["alignment", "test"])[list(metrics([0, 1, 2], np.array([0, 1, 2])))].mean().round(4).to_string())
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
