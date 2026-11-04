import json
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nunchi_ml.accident.predict import AccidentRiskPredictor

def main():
    csv_path = REPO_ROOT / "data" / "expressway" / "segment_1km_dataset_v1.csv"
    df = pd.read_csv(csv_path)
    predictor = AccidentRiskPredictor()
    res = predictor.predict_batch(df)

    points = []
    for _, r in res.iterrows():
        if pd.isna(r["lat"]) or pd.isna(r["lon"]):
            continue
        points.append({
            "route": str(r["route"]),
            "km": int(r["segment_km"]),
            "lat": round(float(r["lat"]), 5),
            "lon": round(float(r["lon"]), 5),
            "risk": float(r["pred_accident_risk"]),
            "grade": str(r["risk_grade"]),
            "glare_h": round(float(r["annual_glare_hours_fwd"]), 1),
            "commute_h": round(float(r["commute_glare_hours"]), 1),
            "tunnel_dist": round(float(r["min_tunnel_dist_km"]), 2),
            "ew": round(float(r["ew_alignment"]), 2),
        })

    out_json = REPO_ROOT / "docs" / "expressway_risk_data.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False)
    print(f"Exported {len(points)} points to {out_json}")

if __name__ == "__main__":
    main()
