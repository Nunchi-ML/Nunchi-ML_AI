"""Audit historical human votes; not a new model or population probability label.

Explicit human files only. Preserve disagreement and abstentions, use the final
driving-view catalogue, and never change original labels or train a classifier.
"""
import argparse
import hashlib
from html import unescape
import json
from pathlib import Path
import re
import shutil

import pandas as pd

PANELS = {
    "daegu": ("glare_mapillary", "label_set.csv", ["성진", "범수"], "drive_pack/label_tool.html"),
    "seoul": ("glare_mapillary_seoul", "label_set.csv", ["성진", "범수", "수영"], "seoul_pack/label_tool.html"),
    "aihub": ("aihub_special", "aihub_label_set.csv", ["성진", "범수", "수영"], "label_pack/label_tool.html"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(path):
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "id" not in frame or frame.id.eq("").any() or not frame.id.is_unique:
        raise ValueError("Missing/duplicate photo ID")
    return frame


def read_votes(catalogue, raw, rater):
    if not raw.id.is_unique or raw.id.eq("").any() or not set(raw.id).issubset(catalogue.id):
        raise ValueError("Duplicate/unknown raw photo ID")
    if set(raw.labeler) != {rater}:
        raise ValueError("Human identity does not match declared panel file")
    reference = catalogue.set_index("id").loc[raw.id]
    for name in ["file", "sequence"]:
        if not (reference[name].to_numpy() == raw[name].to_numpy()).all():
            raise ValueError("Raw image/sequence reference mismatch")
    label = raw.label.str.strip().str.lower()
    if not label.isin(["0", "1", "2", "x", ""]).all():
        raise ValueError("Unknown raw label")
    return pd.DataFrame(dict(id=raw.id.to_numpy(), rater=rater, raw_label=raw.label.to_numpy(),
                             label=label.to_numpy(), unsure=raw.unsure.to_numpy(), note=raw.note.to_numpy()))


def aggregate_votes(catalogue, votes):
    if votes.duplicated(["id", "rater"]).any():
        raise ValueError("Repeated person cannot count as an independent vote")
    if not set(votes.id).issubset(catalogue.id):
        raise ValueError("Unknown photo ID")
    valid = votes.label.isin(["0", "1", "2"])
    if not votes.label.isin(["0", "1", "2", "x", ""]).all():
        raise ValueError("Unknown normalized label")
    annotated = votes.assign(valid=valid, any_glare=valid & votes.label.isin(["1", "2"]),
                             severe=valid & votes.label.eq("2"), abstain=votes.label.eq("x"),
                             missing=votes.label.eq(""), uncertain=valid & votes.unsure.eq("1"))
    counts = annotated.groupby("id", sort=False).agg(n=("valid", "sum"), any_glare_k=("any_glare", "sum"),
                severe_k=("severe", "sum"), abstain_n=("abstain", "sum"), missing_n=("missing", "sum"),
                uncertain_n=("uncertain", "sum"), recorded_people=("rater", "nunique"))
    result = catalogue.merge(counts, on="id", how="left", validate="one_to_one")
    numeric = ["n", "any_glare_k", "severe_k", "abstain_n", "missing_n", "uncertain_n", "recorded_people"]
    result[numeric] = result[numeric].fillna(0).astype(int)
    for event in ["any_glare", "severe"]:
        result["legacy_" + event + "_share"] = result[event + "_k"] / result.n.where(result.n.gt(0))
    return result


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(data):
        raise ValueError("Use a new private output directory")
    here = Path(__file__).resolve().parent
    final_view = here.parent / "glare_video_source/sequence_view.csv"
    old_view = data / "glare_mapillary/sequence_view.csv"
    cache = data / "road_visibility_20261104/run30/training_features.csv"
    files = [Path(__file__), final_view, old_view, cache]
    for folder, name, raters, tool in PANELS.values():
        files.extend([data / folder / name, data / folder / tool, data / folder / "labels_merged.csv"])
        files.extend(data / folder / f"labels_{r}.csv" for r in raters)
    hashes = {str(path): digest(path) for path in files}
    out.mkdir()
    specification = dict(status="frozen_before_audit", panels=PANELS,
        inclusion="Daegu final source-code road/dashcam view; Seoul and AIHub full historical catalogues",
        events="Report both legacy grade>=1 and grade==2 without choosing a new meaning of high",
        exclusions="No AI votes, single-rater newcamera50, repeats, relabeling, model fitting, probability/score claims",
        limitation="Historical 2/3-person convenience panels; viewing conditions and compliance with independent rating unverified",
        hashes=hashes)
    (out / "specification_before_audit.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, out / "source_before_audit.py")
    cached = pd.read_csv(cache, dtype={"id": str})[["id", "group", "camera", "image_sha256"]]
    if not cached.id.is_unique:
        raise ValueError("Duplicate cached IDs")
    tables, long_rows, summaries, rubric_rows, image_hashes = [], [], [], [], {}
    for panel, (folder, name, raters, tool) in PANELS.items():
        catalogue = read_table(data / folder / name)
        catalogue["eligible"] = True
        if panel == "daegu":
            final, old = pd.read_csv(final_view, dtype=str), pd.read_csv(old_view, dtype=str)
            if not final.sequence.is_unique or not old.sequence.is_unique:
                raise ValueError("Ambiguous sequence-view mapping")
            catalogue = catalogue.merge(final[["sequence", "view"]].rename(columns={"view": "final_view"}), on="sequence", how="left", validate="many_to_one")
            catalogue = catalogue.merge(old[["sequence", "view"]].rename(columns={"view": "older_view"}), on="sequence", how="left", validate="many_to_one")
            if catalogue[["final_view", "older_view"]].isna().any().any():
                raise ValueError("Missing sequence-view decision")
            catalogue["eligible"] = catalogue.final_view.isin(["road", "dashcam"])
            catalogue["older_eligible"] = catalogue.older_view.isin(["road", "dashcam"])
            known = read_table(data / folder / "labels_merged.csv")
            if set(catalogue.loc[catalogue.eligible, "id"]) != set(known.id):
                raise ValueError("Final driving catalogue no longer matches historical124")
        votes = pd.concat([read_votes(catalogue, read_table(data / folder / f"labels_{r}.csv"), r) for r in raters], ignore_index=True)
        votes["panel"] = panel
        long_rows.append(votes)
        result = aggregate_votes(catalogue, votes)
        result["panel"] = panel
        result["declared_panel_size"] = len(raters)
        part = result[result.eligible].copy()
        for row in part.to_dict("records"):
            path = (data / folder / row["file"]).resolve()
            if not path.is_relative_to(data / folder):
                raise ValueError("Image path escaped panel directory")
            image_hashes[str(path)] = digest(path)
            part.loc[part.id.eq(row["id"]), "image_sha256"] = image_hashes[str(path)]
        if not part.n.eq(len(raters)).all():
            raise ValueError("Historical eligible panel is incomplete; report missing votes before use")
        matched = part.merge(cached, on="id", how="left", validate="one_to_one", suffixes=("", "_cache"))
        available = matched.image_sha256_cache.notna()
        if not matched.loc[available, "image_sha256"].eq(matched.loc[available, "image_sha256_cache"]).all():
            raise ValueError("Cached photo bytes do not match original panel image")
        matched["existing_feature_row"] = available
        matched["cached_capture_group"] = matched["group_cache"]
        tables.append(matched)
        result.to_csv(out / f"{panel}_all_catalogue_audit.csv", index=False, encoding="utf-8-sig")
        text = (data / folder / tool).read_text(encoding="utf-8")
        rubric = re.search(r"<table>(.*?)</table>", text, re.S)
        if not rubric:
            raise ValueError("Current tool rubric missing")
        normalized = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", rubric[1]))).strip()
        rubric_rows.append(dict(panel=panel, current_tool_rubric=normalized,
            independence_requested="서로 상의하지 않고" in text,
            historical_display_conditions="unknown", historical_compliance="not verifiable from CSV"))
        summaries.append(dict(panel=panel, catalogue_rows=len(catalogue), eligible_photos=len(part),
            valid_votes=int(part.n.sum()), raters=len(raters), feature_rows=int(available.sum()),
            missing_feature_rows=int((~available).sum()), any_glare_mixed=int(((part.any_glare_k > 0) & (part.any_glare_k < part.n)).sum()),
            severe_mixed=int(((part.severe_k > 0) & (part.severe_k < part.n)).sum()),
            severe_positive_photos=int(part.severe_k.gt(0).sum()),
            older_view_only=int((result.get("older_eligible", result.eligible) & ~result.eligible).sum())))
    canonical = pd.concat(tables, ignore_index=True)
    if not canonical.id.is_unique:
        raise ValueError("Same photo appears in multiple panels")
    canonical.to_csv(out / "legacy_panel_shares.csv", index=False, encoding="utf-8-sig")
    pd.concat(long_rows, ignore_index=True).to_csv(out / "raw_votes_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rubric_rows).to_csv(out / "rubric_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    if any(digest(path) != value for path, value in (hashes | image_hashes).items()):
        raise ValueError("Source/votes/images changed during audit")
    result = dict(status="complete", panels=summaries, eligible_photos=len(canonical), valid_votes=int(canonical.n.sum()),
        unique_self_reported_people=len(set(sum([p[2] for p in PANELS.values()], []))),
        existing_feature_rows=int(canonical.existing_feature_row.sum()),
        duplicate_image_bytes=int(canonical.image_sha256.duplicated().sum()),
        rubrics_currently_identical=len({r["current_tool_rubric"] for r in rubric_rows}) == 1,
        readiness="Historical panel proportions recoverable; new human-population high-rating target not adopted or validated",
        no_models_trained=True, hashes=hashes, image_hashes=image_hashes)
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Historical human vote audit</title><h1>Historical panel proportions only</h1><p>No model trained. These are convenience-panel historical grade shares, not general population probabilities or new-target measurements.</p>'+pd.DataFrame(summaries).to_html(index=False)+'<h2>Current tool instructions; historical display conditions unknown</h2>'+pd.DataFrame(rubric_rows).to_html(index=False), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("Recovered", len(canonical), "photos /", int(canonical.n.sum()), "valid votes; no models trained")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
