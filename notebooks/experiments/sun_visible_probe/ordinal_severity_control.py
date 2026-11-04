"""Keep the existing G-only severity input while changing to true cumulative targets.

This is an additional logistic control, not the former conditional two-stage
model: B learns y>=2 from ALL training labels, A learns y>=1. Both outputs
are projected before argmax. Feature selection/scaling are inside each fit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ordinal_compare as O
from nunchi_ml.glare.ordinal import CumulativeOrdinalClassifier, cumulative_to_class_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to(Path("data").resolve()):
        parser.error("Use a new output directory within private data")
    args.out.mkdir(parents=True)
    data = pd.read_csv(args.manifest, dtype={"id": str})
    train = data[data.camera.ne("AI Hub")].reset_index(drop=True)
    cols = O.SETS["g_rel"]
    scores, binary, diagnostics, predictions = [], [], [], []
    for seed in range(30):
        store = {}
        for tr, va in GroupKFold(5, shuffle=True, random_state=seed).split(train, train.label, train.group):
            assert not set(train.iloc[tr].group) & set(train.iloc[va].group)
            for a_weight, b_weight in [("balanced", "balanced"), ("balanced", None), (None, "balanced")]:
                name = f"ordinal_severity_G/A_{a_weight}/B_{b_weight}"
                A = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight=a_weight))
                B = make_pipeline(ColumnTransformer([("G", "passthrough", list(range(len(O.G_COLUMNS))))]),
                                  StandardScaler(), LogisticRegression(max_iter=3000, class_weight=b_weight))
                model = CumulativeOrdinalClassifier(A, B).fit(train.iloc[tr][cols], train.iloc[tr].label)
                q = model.predict_cumulative_proba(train.iloc[va][cols], corrected=False)
                store.setdefault(name, {}).update(dict(zip(va, q)))
        y = train.label.to_numpy(int)
        for name, values in store.items():
            q = np.stack([values[i] for i in range(len(train))])
            p = cumulative_to_class_probabilities(q)
            pred = p.argmax(1)
            meta = dict(test="group_cv", repeat=seed, method=name)
            scores.append({**meta, **O.metrics(y, pred)})
            binary.append({**meta, "decision": "direct_binary_A", **O.binary_metrics(y, q[:, 0] >= .5, q[:, 0])})
            diagnostics.append({**meta, "violation_rate": float((q[:, 1] > q[:, 0]).mean())})
            predictions.extend({**meta, "id": d.id, "label": int(d.label), "pred": int(pred[i]),
                                "raw_q1": float(q[i, 0]), "raw_q2": float(q[i, 1]),
                                **{f"p{k}": float(p[i, k]) for k in range(3)}} for i, d in enumerate(train.itertuples()))
        print(f"severity control repeat={seed} done", flush=True)
    for filename, rows in [("scores.csv", scores), ("binary_scores.csv", binary),
                           ("threshold_scores.csv", diagnostics), ("predictions.csv", predictions)]:
        pd.DataFrame(rows).to_csv(args.out / filename, index=False, encoding="utf-8-sig")
    info = dict(status="complete", input_sha256=O.fingerprint(args.manifest),
                sources={str(p): O.fingerprint(p) for p in [Path(__file__), Path(O.__file__),
                         Path(O.sys.modules[CumulativeOrdinalClassifier.__module__].__file__)]},
                features_A=cols, features_B=O.G_COLUMNS, repeats=30,
                limitation="Development control following initial failures, not independent confirmation. No final model decision/export.")
    (args.out / "method.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(scores).groupby("method")[["macro_f1", "accuracy", "qwk", "mae", "f1_1"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
