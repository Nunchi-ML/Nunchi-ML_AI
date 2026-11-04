import copy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("gemini_reference_probe")


@pytest.fixture
def packet(module, tmp_path):
    images = {str((tmp_path / f"C{i:03d}_+0.jpg").resolve()): "hash" for i in range(1, 6)}
    packet = dict(cwd=str(tmp_path), condition="center", case_ids=[f"C{i:03d}" for i in range(1, 6)], expected_images=images)
    items = [dict(case_id=cid, pixels_viewed=True, scene="forward_or_aligned_road", reference_grade="1",
                  used_offsets=[0], reference_comparability="not_provided", evidence="visible sun; road remains", uncertainty="no human vision measurement")
             for cid in packet["case_ids"]]
    events = [dict(event="step_update", step_update=dict(tool_name="read_file", state="DONE", tool_info=dict(parameters=dict(Path=p)))) for p in images]
    events += [dict(event="result", result=dict(status="SUCCESS", structured_output=dict(items=items), conversation_id="fresh-case-batch"))]
    return packet, events


def test_actual_reads_and_case_coverage_required(module, packet):
    p, events = packet
    result = module.validate_packet(p, events)
    assert result["image_reads"] == 5 and len(result["items"]) == 5


def test_provider_schema_avoids_broken_numeric_enum_but_checks_range(module, packet):
    array_schema = module.schema(["C001"])["properties"]["items"]["items"]["properties"]["used_offsets"]
    assert array_schema["items"] == {"type": "integer"}
    p, events = packet
    events[-1]["result"]["structured_output"]["items"][0]["used_offsets"] = [0, 99]
    with pytest.raises(ValueError, match="offsets"):
        module.validate_packet(p, events)


@pytest.mark.parametrize("error", ["missing_case", "duplicate_case", "missing_read", "outside_read", "command", "future", "out_of_scope", "grade", "failure"])
def test_unblinded_or_incomplete_model_response_is_rejected(module, packet, error):
    p, events = copy.deepcopy(packet)
    items = events[-1]["result"]["structured_output"]["items"]
    if error == "missing_case":
        items.pop()
    elif error == "duplicate_case":
        items[1]["case_id"] = items[0]["case_id"]
    elif error == "missing_read":
        events.pop(0)
    elif error == "outside_read":
        events[0]["step_update"]["tool_info"]["parameters"]["Path"] = str(Path(p["cwd"]).parent / "answers.csv")
    elif error == "command":
        events[0]["step_update"]["tool_name"] = "run_command"
    elif error == "future":
        items[0]["used_offsets"] = [0, 1]
    elif error == "out_of_scope":
        items[0]["scene"] = "rear_road"
    elif error == "grade":
        items[0]["reference_grade"] = "3"
    else:
        events[-1]["result"]["status"] = "ERROR"
    with pytest.raises(ValueError):
        module.validate_packet(p, events)


def test_prepare_keeps_original_bytes_answers_outside_and_separates_conditions(module, tmp_path):
    root = tmp_path / "temporal"
    root.mkdir()
    source = root / "original_labels.csv"
    source.write_text("SECRET-ANSWER-DO-NOT-INCLUDE", encoding="utf-8")
    module.dump(root / "method.json", dict(status="complete", rows=50, input_sha256={str(source): module.sha(source)}))
    for i in range(1, 51):
        directory = root / "review" / f"T{i:03d}"
        directory.mkdir(parents=True)
        for offset in module.OFFSETS:
            name = "center.jpg" if offset == 0 else f"neighbor_{offset:+.0f}.jpg"
            (directory / name).write_bytes(f"unchanged pixels {i} {offset}".encode())
    out, inputs = tmp_path / "private", tmp_path / "blind_inputs"
    manifest = module.prepare(root, out, inputs)
    assert len(manifest["packets"]) == 20
    assert [len(p["expected_images"]) for p in manifest["packets"][:2]] == [5, 25]
    assert len(json.loads((out / "private_pairs.json").read_text(encoding="utf-8"))) == 100
    for packet in manifest["packets"]:
        text = (Path(packet["cwd"]) / "prompt.txt").read_text(encoding="utf-8")
        assert "SECRET-ANSWER" not in text and "T001" not in text
        assert all(module.sha(path) == digest for path, digest in packet["expected_images"].items())
    assert source.read_text(encoding="utf-8") == "SECRET-ANSWER-DO-NOT-INCLUDE"
    with pytest.raises(ValueError, match="new outputs"):
        module.prepare(root, out, inputs)
