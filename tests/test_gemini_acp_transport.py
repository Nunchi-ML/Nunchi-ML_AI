import copy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def example(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    module = importlib.import_module("gemini_acp_transport")
    ids = [f"C{i:03d}" for i in range(1, 6)]
    packet = dict(condition="center", case_ids=ids, prompt_sha256="prompt-digest", expected_images={str(tmp_path / f"{cid}.jpg"): "digest" for cid in ids})
    items = [dict(case_id=cid, pixels_viewed=True, scene="forward_or_aligned_road", sun_evidence="yes",
                  lane="identifiable", vehicle="not_evaluable", sign="not_evaluable", reference_grade="1",
                  reference_comparability="not_provided", used_offsets=[0], evidence="road marking visible", uncertainty="photo-only evidence") for cid in ids]
    events = [dict(id=0, result=dict(agentCapabilities=dict(promptCapabilities=dict(image=True)))),
              dict(id=1, result=dict(sessionId="new-session", models=dict(currentModelId=module.MODEL))),
              dict(event="client_attachment_delivery", images=packet["expected_images"], prompt_sha256="prompt-digest"),
              dict(method="session/update", params=dict(sessionId="new-session", update=dict(sessionUpdate="agent_message_chunk", content=dict(type="text", text=json.dumps(dict(items=items)))))),
              dict(id=2, result=dict(stopReason="end_turn"))]
    return module, packet, events, items


def test_direct_attachments_completed_model_response(example):
    module, packet, events, _ = example
    valid = module.validate_packet(packet, events)
    assert valid["image_attachments"] == 5 and len(valid["items"]) == 5


def test_temporal_reference_offsets_do_not_claim_proof_of_center_attention(example):
    module, packet, events, items = example
    packet["condition"] = "temporal"
    for item in items:
        item["used_offsets"] = [-2, -1, 1, 2]
        item["reference_comparability"] = "consistent_scene"
    events[3]["params"]["update"]["content"]["text"] = json.dumps(dict(items=items))
    valid = module.validate_packet(packet, events)
    assert valid["items"][0]["used_offsets"] == [-2, -1, 1, 2]
    assert "not proof" in valid["offset_semantics"]


@pytest.mark.parametrize("contaminated", [False, True])
def test_resume_reaudits_raw_turns_and_never_retries_existing_batches(example, tmp_path, monkeypatch, contaminated):
    transport, packet, events, _ = example
    probe = importlib.import_module("gemini_reference_probe")
    out = tmp_path / "experiment"
    completed = out / "runs" / "center_01"
    completed.mkdir(parents=True)
    packets = [dict(packet, name=f"center_{i:02d}") for i in range(1, 21)]
    if contaminated:
        events.append(dict(params=dict(update=dict(sessionUpdate="tool_call"))))
    raw = "\n".join(json.dumps(e) for e in events)
    (completed / "stdout.jsonl").write_text(raw, encoding="utf-8")
    probe.dump(out / "manifest.json", dict(transport="acp_images", packets=packets, source_sha256={}))
    probe.dump(out / "progress.json", dict(status="incomplete", packets={"center_01": dict(valid=False)}))
    calls = []

    def fake_run(packet, output, agy):
        calls.append(packet["name"])
        return dict(valid=True, cases=5)

    monkeypatch.setattr(transport, "run_one", fake_run)
    if contaminated:
        with pytest.raises(ValueError, match="tool"):
            probe.run(out, "fake-cli", resume=True)
        assert not calls
    else:
        result = probe.run(out, "fake-cli", resume=True)
        assert len(calls) == 19 and "center_01" not in calls and len(result) == 20
        assert (completed / "stdout.jsonl").read_text(encoding="utf-8") == raw
        assert len(list(out.glob("progress_before_resume_*.json"))) == 1


@pytest.mark.parametrize("fault", ["no_image_support", "wrong_model", "missing_attachment", "partial_turn", "tool", "cross_session", "permission", "duplicate", "bad_enum", "boolean_offset", "future", "scope"])
def test_rejects_incomplete_or_contaminated_judgments(example, fault):
    module = example[0]
    packet, events, items = copy.deepcopy(example[1:])
    if fault == "no_image_support":
        events[0]["result"]["agentCapabilities"]["promptCapabilities"]["image"] = False
    elif fault == "wrong_model":
        events[1]["result"]["models"]["currentModelId"] = "other-model"
    elif fault == "missing_attachment":
        events[2]["images"] = {}
    elif fault == "partial_turn":
        events[-1]["result"]["stopReason"] = "cancelled"
    elif fault == "tool":
        events.append(dict(params=dict(update=dict(sessionUpdate="tool_call", title="read answers"))))
    elif fault == "cross_session":
        events[3]["params"]["sessionId"] = "other-session"
    elif fault == "permission":
        events.append(dict(method="session/request_permission"))
    else:
        if fault == "duplicate":
            items[1]["case_id"] = items[0]["case_id"]
        elif fault == "bad_enum":
            items[0]["lane"] = "invented-state"
        elif fault == "boolean_offset":
            items[0]["used_offsets"] = [False]
        elif fault == "future":
            items[0]["used_offsets"] = [0, 1]
        elif fault == "scope":
            items[0]["scene"] = "rear_road"
        events[3]["params"]["update"]["content"]["text"] = json.dumps(dict(items=items))
    with pytest.raises(ValueError):
        module.validate_packet(packet, events)
