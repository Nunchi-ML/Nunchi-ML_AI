import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("sd_frame_curation")


def test_path_traversal_and_invalid_clock_are_not_trusted(module, tmp_path):
    assert module.safe_image(tmp_path, "images/a.jpg") == tmp_path/"images/a.jpg"
    with pytest.raises(ValueError, match="inside root/images"):
        module.safe_image(tmp_path, "images/../../outside.jpg")
    assert module.clip_time("invalid", 0) is None


def test_exact_duplicates_prefer_driving_and_link_existing_reference(module):
    frame = pd.DataFrame(dict(image=["images/p", "images/i", "images/e", "images/x"], decode_ok=[True]*4,
                              source_category=["PARK", "INF", "EVENT", "USER"], sha256=["same", "same", "other", "old"]))
    out = module.exact_representatives(frame, {"old"})
    assert out.set_index("image").loc["images/p", "exact_representative"] == "images/i"
    assert out.set_index("image").loc["images/p", "exact_duplicate"]
    assert out.set_index("image").loc["images/x", "existing_reference_duplicate"]


def test_dark_blurred_and_brightness_transition_cases_are_protected(module):
    frame = pd.DataFrame(dict(image=[f"images/{i}.jpg" for i in range(12)], clip_id=["a"]*12, second=range(12),
        decode_ok=[True]*12, source_category=["INF"]*12, mean_brightness=[10]*5+[55]*7,
        white_ratio=[0]*7+[.2]+[0]*4, road_laplacian_var=[100]*8+[2]+[100]*3))
    transitions = pd.DataFrame([dict(clip_id="a", second=5, before_image="images/4.jpg", after_image="images/5.jpg")])
    reasons = module.candidate_reasons(frame, transitions, 10)
    assert "temporal_representative" in reasons["images/0.jpg"]
    assert "transition_context" in reasons["images/4.jpg"]
    assert "transition_context" in reasons["images/6.jpg"]
    assert "high_white_review" in reasons["images/7.jpg"]
    assert "low_sharpness_review" in reasons["images/8.jpg"]


def near_rows():
    return pd.DataFrame([dict(image=f"images/{i}.jpg", clip_id="clip", source_category="INF", group_id="20261105",
        second=i, filename_datetime=pd.Timestamp("2026-11-05 08:00:00")+pd.Timedelta(seconds=i),
        decode_ok=True, existing_reference_duplicate=False, exact_duplicate=False, phash="0000000000000000", dhash="0000000000000000") for i in range(3)])


def test_near_duplicates_do_not_discard_protected_or_distant_unknown_frames(module):
    frame = near_rows(); thumbs = {s:np.full((18,32), 100, np.uint8) for s in frame.image}
    reasons = {"images/0.jpg":{"transition_context"}, "images/1.jpg":{"temporal_representative"}, "images/2.jpg":{"high_white_review"}}
    out, matches = module.thin_candidates(frame, thumbs, reasons)
    values = out.set_index("image").decision.to_dict()
    assert values["images/1.jpg"] == "reserve_near_duplicate"
    assert values["images/0.jpg"] == values["images/2.jpg"] == "retain_candidate"
    assert len(matches) == 2
    a, b = frame.iloc[0].to_dict(), frame.iloc[1].to_dict()
    b["filename_datetime"] = pd.NaT
    assert not module.is_near(a, b, thumbs[a["image"]], thumbs[b["image"]])[0]
    b["filename_datetime"] = pd.Timestamp("2026-11-05 09:00:00")
    assert not module.is_near(a, b, thumbs[a["image"]], thumbs[b["image"]])[0]


def test_review_batch_is_diverse_deterministic_and_label_independent(module):
    frame = pd.DataFrame([dict(image=f"images/{i}.jpg", clip_id=f"c{i//4}", source_category="INF",
        group_id=str(i%3), mean_brightness=float(20+i%150), glare_label_final=i%3) for i in range(80)])
    a = module.review_batch(frame, 20)
    b = module.review_batch(frame.assign(glare_label_final=2), 20)
    assert len(a) == 20 and a.image.is_unique and a.clip_id.nunique() == 20
    assert a.image.tolist() == b.image.tolist()


def test_end_to_end_preserves_source_bytes_labels_and_portable_review(module, tmp_path):
    root, data = tmp_path/"source", tmp_path/"private_data"
    root.mkdir(); data.mkdir(); rows=[]
    for category, date in [("INF", "20261105"), ("PARK", "20261104")]:
        clip=date+"_080000_INF_2"; folder=root/"images"/category/clip; folder.mkdir(parents=True)
        for second in range(12):
            image=np.random.default_rng(second).integers(0,256,(64,64,3),dtype=np.uint8)
            payload=cv2.imencode(".jpg",image)[1].tobytes(); path=folder/f"{second+1:06d}.jpg";path.write_bytes(payload)
            rows.append(dict(image=path.relative_to(root).as_posix(),clip_id=clip,second=second,group_id=date,
                source_category=category,camera="front",mean_brightness=100,white_ratio=.01,
                **{label:"" for label in module.LABELS}))
    frame=pd.DataFrame(rows);frame.to_csv(root/"frames_and_labels.csv",index=False)
    pd.DataFrame(dict(file=[c+".avi" for c in frame.clip_id.unique()])).to_csv(root/"video_inventory.csv",index=False)
    pd.DataFrame(columns=["clip_id","second","before_image","after_image"]).to_csv(root/"model3_brightness_transition_candidates.csv",index=False)
    (root/"README.txt").write_text("source");(root/"summary.json").write_text("{}")
    ref=data/"road_visibility_20261104/run30";ref.mkdir(parents=True)
    pd.DataFrame(dict(image_sha256=["old"])).to_csv(ref/"training_features.csv",index=False)
    hashes={str(p):module.sha(p) for p in root.rglob("*") if p.is_file()}
    out=data/"curated"
    report=module.run(root,data,out,workers=1,review_size=3)
    assert report["source_files_deleted"] == 0 and not report["model_training"]
    assert report["review_photos"] == 3 and report["parking_photos"] == 12
    assert all(module.sha(p)==value for p,value in hashes.items())
    labels=pd.read_csv(out/"review_first300/labels_blank.csv")
    assert labels.glare_label_final.isna().all()
    assert len(list((out/"review_first300/images").glob("*.jpg"))) == 3
    text=(out/"review_first300/review.html").read_text(encoding="utf-8")
    assert "const rows=" in text and "images/sd_" in text
    decisions=pd.read_csv(out/"all_frame_decisions.csv")
    assert decisions.image.nunique() == 24
    with pytest.raises(ValueError,match="new private"):
        module.run(root,data,out)
