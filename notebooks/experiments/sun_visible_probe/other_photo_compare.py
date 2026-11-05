"""Extend the frozen photo-method ablation outside human50, with AI references.

Reproduce every human50 prediction before reporting. Fit only the same624 rows,
never the new evaluation labels. Existing pack/videos are reused development
data; this is neither human accuracy nor a new independent test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd

import road_visibility_probe as R
from new_camera_audit import load_labels
from nunchi_ml.glare.features import extract_array
from rel_features import rel_array
from ordinal_compare import metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition_ids(sheet, human, accepted):
    sheet, human, accepted = [pd.Index(x) for x in [sheet, human, accepted]]
    if any(x.hasnans or not x.is_unique for x in [sheet, human, accepted]):
        raise ValueError("IDs must be unique/present")
    if len(human.difference(sheet)) or len(accepted.difference(sheet)):
        raise ValueError("Label IDs outside original pack")
    outside = sheet.difference(human, sort=False)
    return outside, outside.intersection(accepted, sort=False), outside.difference(accepted, sort=False)


def verify_reproduction(actual, expected):
    keys = ["id", "method"]
    if any(frame.duplicated(keys).any() for frame in [actual, expected]):
        raise ValueError("Duplicate frozen prediction key")
    a = actual.set_index(keys).pred.sort_index()
    b = expected.set_index(keys).pred.sort_index()
    if not a.index.equals(b.index) or not np.array_equal(a.to_numpy(), b.to_numpy()):
        raise ValueError("Human50 predictions differ from frozen ablation")


def run(data, root, out):
    data, root, out = (Path(p).resolve() for p in [data, root, out])
    if out.exists() or out.parent != root or not root.is_relative_to(data):
        raise ValueError("Use new private output folder")
    protocol_files = [root / "protocol_before_analysis.json", root / "extension_protocol_before_fit.json"]
    protocols = [json.loads(p.read_text(encoding="utf-8")) for p in protocol_files]
    fingerprints = {str(p): sha(p) for p in protocol_files}
    for protocol in protocols:
        for path, digest in protocol["source_sha256"].items():
            if sha(path) != digest:
                raise ValueError("Frozen inputs/source changed")
            fingerprints[path] = digest
    pack, previous = data / "new_cameras", data / "road_visibility_20261104/run30"
    tr = pd.read_csv(previous / "training_features.csv", dtype={"id": str})
    human = pd.read_csv(previous / "human_features.csv", dtype={"id": str})
    if len(tr) != 624 or len(human) != 50 or not tr.id.is_unique or not human.id.is_unique:
        raise ValueError("Require preserved624 training rows and50 reproduction rows")
    sheet = pd.read_csv(pack / "label_pack/label_sheet_blind.csv", dtype={"id": str})[["id", "file"]]
    original_human = pd.read_csv(pack / "human_review_50/labels_성진.csv", dtype={"id": str})
    if set(human.id) != set(original_human.id):
        raise ValueError("Human reproduction set changed")
    labels, _, info = load_labels([pack / "labels_claude.csv"], sheet.id, "ai_provisional")
    outside, accepted, pending = partition_ids(sheet.id, human.id, labels.index)
    if (len(outside), len(accepted), len(pending)) != (299, 277, 22):
        raise ValueError("Frozen outside-photo coverage changed")
    fmap = pd.read_csv(pack / "private_frame_map.csv", dtype={"id": str})[["id", "video"]]
    te = sheet[sheet.id.isin(accepted)].merge(labels.rename("label"), left_on="id", right_index=True, validate="one_to_one").merge(fmap, on="id", validate="one_to_one")
    if len(te) != 277 or te.video.isna().any():
        raise ValueError("Every accepted image needs a video and label")
    paths, extracted = [], []
    image_root = (pack / "label_pack").resolve()
    for row in te.itertuples():
        path = (image_root / row.file).resolve()
        if not path.is_relative_to(image_root):
            raise ValueError("Image outside original pack")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in set(tr.image_sha256) or digest in set(human.image_sha256):
            raise ValueError("Photo byte overlap with training/human50")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        extracted.append(dict(image_sha256=digest, **extract_array(image), **{"r_"+k: v for k, v in rel_array(image).items()}))
        paths.append(path)
        fingerprints[str(path)] = digest
    te = pd.concat([te.reset_index(drop=True), pd.DataFrame(extracted)], axis=1)
    if te.image_sha256.duplicated().any():
        raise ValueError("Duplicate evaluation image bytes")
    old = pd.read_csv(pack / "audit_20261104/predictions.csv", dtype={"id": str}).set_index("id").loc[te.id]
    if not np.array_equal(old.image_sha256.to_numpy(), te.image_sha256) or not np.array_equal(old.label.to_numpy(), te.label):
        raise ValueError("Old prediction image/reference mismatch")
    out.mkdir()
    te = R.attach_road(te, paths, out / "road_features.csv")
    all_test = pd.concat([human.assign(evaluation="human50_reproduction"), te.assign(evaluation="outside277_AI_provisional")], ignore_index=True)
    predictions = {}
    for name, (cols, family) in R.PHOTO_METHODS.items():
        if not np.isfinite(tr[cols].to_numpy(float)).all() or not np.isfinite(all_test[cols].to_numpy(float)).all():
            raise ValueError("Missing/nonfinite features; no row dropping")
        if family == "lightgbm":
            model = R.lightgbm_model().fit(tr[cols], tr.label)
            p = np.zeros((len(all_test), 3))
            p[:, model.classes_] = model.predict_proba(all_test[cols])
        else:
            p = R.C.probabilities(tr, all_test, cols)
        R.validate_probabilities(p)
        predictions[name] = p.argmax(1) if family == "lightgbm" else R.C.decision(p)
        print(name, "fit624/predict327 complete", flush=True)
    predictions["always_normal"] = np.zeros(len(all_test), int)
    records = [dict(id=row.id, method=name, pred=int(pred[i])) for name, pred in predictions.items() for i, row in enumerate(all_test.itertuples()) if row.evaluation == "human50_reproduction"]
    frozen = pd.read_csv(previous / "predictions.csv", dtype={"id": str})
    frozen = frozen[frozen.test.eq("human50_photo_only")][["id", "method", "pred"]]
    verify_reproduction(pd.DataFrame(records), frozen)
    # Also compare baseline predictions to their earlier all-camera execution.
    for name, column in [("screen_lgb", "ML1 LightGBM(화면 12, 기존 모델 ② 형태)"), ("rel_logistic", "ML2 로지스틱(상대 12)")]:
        if not np.array_equal(predictions[name][50:], old[column].to_numpy()):
            raise ValueError("Outside baseline differs from earlier fixed model")
    scores, per_video = [], []
    for regime in all_test.evaluation.unique():
        mask = all_test.evaluation.eq(regime).to_numpy()
        subset = all_test[mask]
        for name, values in predictions.items():
            scores.append(dict(evaluation=regime, method=name, rows=len(subset), **metrics(subset.label.to_numpy(), values[mask])))
        if regime.startswith("outside"):
            for video in subset.video.unique():
                local = mask & all_test.video.eq(video).to_numpy()
                for name, values in predictions.items():
                    per_video.append(dict(video=video, method=name, rows=int(local.sum()), **metrics(all_test.loc[local, "label"].to_numpy(), values[local])))
    outside_predictions = {name: values[50:] for name, values in predictions.items()}
    bootstrap = R.grouped_bootstrap(te, outside_predictions, protocols[1]["bootstrap"])
    pd.DataFrame(scores).to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_video).to_csv(out / "per_video.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bootstrap).to_csv(out / "paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    result_predictions = all_test[["id", "file", "video", "label", "image_sha256", "evaluation"]].copy()
    for name, values in predictions.items():
        result_predictions[name] = values
    result_predictions.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    source_files = [Path(__file__), Path(R.__file__), Path(R.R.__file__), Path(R.C.__file__), Path(R.P.__file__)]
    result = dict(status="complete", train_rows=624, reproduction_rows=50, exact_reproduced_predictions=len(records),
                  total_outside=299, accepted_AI=277, pending_AI=22, AI_class_counts=te.label.value_counts().sort_index().to_dict(),
                  videos=int(te.video.nunique()), shared_videos_with_human50=len(set(te.video) & set(human.video)),
                  methods=R.PHOTO_METHODS, scores=scores, paired_bootstrap=bootstrap, label_evidence=info,
                  input_sha256=fingerprints, source_sha256={str(p): sha(p) for p in source_files},
                  limitations=["Existing AI-provisional reference, not human ground truth or new labels",
                               "Reused pack and video groups; not a new independent test",
                               "Same624 training and methods; no threshold/model selection or calibration",
                               "Only7 severe AI labels, camera independence unverified",
                               "No G inputs, deployment change, final selection, or direct .6229 improvement claim"])
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, out / Path(__file__).name)
    (out / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Outside human50 photo comparison</title><style>body{font:16px system-ui;margin:30px}td,th{padding:6px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Outside human50: reused AI-reference development comparison</h1><p>299 outside photos:277 accepted/22 abstained; same624 training, no tuning or label changes. All300 human50 predictions reproduced. Not human accuracy, independent test or improvement over original320 CV .6229.</p>' + pd.DataFrame(scores).round(4).to_html(index=False) + '<h2>Paired video bootstrap, fixed seed</h2>' + pd.DataFrame(bootstrap).round(4).to_html(index=False), encoding="utf-8")
    if any(sha(p) != value for p, value in fingerprints.items()):
        raise ValueError("Inputs changed during comparison")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--root", type=Path, default=Path("data/other_photos_20261105"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data, args.root, args.out)
    print(json.dumps({k: result[k] for k in ["status", "exact_reproduced_predictions", "accepted_AI", "pending_AI", "paired_bootstrap"]}, ensure_ascii=False))
