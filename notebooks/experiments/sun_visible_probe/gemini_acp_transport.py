"""Official ACP image blocks, cached CLI login, fresh isolated sessions.

No client filesystem/terminal capabilities or permission approvals are offered.
Attachment delivery is logged; model claims of viewing are not proof of attention.
"""
import base64
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

from gemini_reference_probe import MODEL, OFFSETS, dump, schema, sha


def check_schema(value, spec):
    """Validate the small frozen output schema without an extra dependency."""
    kind = spec["type"]
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": isinstance(value, bool),
             "integer": isinstance(value, int) and not isinstance(value, bool)}[kind]
    if not valid or ("enum" in spec and value not in spec["enum"]):
        raise ValueError("Response schema type/enum mismatch")
    if kind == "object":
        if set(value) != set(spec["required"]):
            raise ValueError("Response schema missing/extra fields")
        for key, child in value.items():
            check_schema(child, spec["properties"][key])
    if kind == "array":
        if not spec.get("minItems", 0) <= len(value) <= spec.get("maxItems", float("inf")):
            raise ValueError("Response schema array size")
        for child in value:
            check_schema(child, spec["items"])


def validate_packet(packet, events):
    initial = [e["result"] for e in events if e.get("id") == 0 and "result" in e]
    sessions = [e["result"] for e in events if e.get("id") == 1 and "result" in e]
    finals = [e["result"] for e in events if e.get("id") == 2 and "result" in e]
    deliveries = [e for e in events if e.get("event") == "client_attachment_delivery"]
    if (len(initial) != 1 or not initial[0].get("agentCapabilities", {}).get("promptCapabilities", {}).get("image")
            or len(sessions) != 1 or sessions[0].get("models", {}).get("currentModelId") != MODEL
            or len(finals) != 1 or finals[0].get("stopReason") != "end_turn"):
        raise ValueError("Image-capable pinned-model session and completed turn required")
    if (len(deliveries) != 1 or deliveries[0]["images"] != packet["expected_images"]
            or deliveries[0].get("prompt_sha256") != packet["prompt_sha256"]):
        raise ValueError("Incomplete/changed image attachments")
    chunks = []
    session = sessions[0]["sessionId"]
    for event in events:
        if "error" in event:
            raise ValueError("ACP returned error")
        if event.get("method") == "session/request_permission":
            raise ValueError("Unexpected permission request")
        update = event.get("params", {}).get("update", {})
        if update.get("sessionUpdate") in {"tool_call", "tool_call_update"}:
            raise ValueError("Unexpected tool/context access")
        if event.get("method") == "session/update" and event["params"].get("sessionId") != session:
            raise ValueError("Cross-session content")
        if update.get("sessionUpdate") == "agent_message_chunk":
            if update.get("content", {}).get("type") != "text":
                raise ValueError("Unexpected response content")
            chunks.append(update["content"]["text"])
    text = "".join(chunks).strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[len("```json\n"):-3].strip()
    result = json.loads(text)
    check_schema(result, schema(packet["case_ids"]))
    items = result["items"]
    if {x["case_id"] for x in items} != set(packet["case_ids"]):
        raise ValueError("Incomplete/duplicate case coverage")
    for item in items:
        offsets = item["used_offsets"]
        if (not item["pixels_viewed"] or not item["evidence"].strip() or not item["uncertainty"].strip()
                or len(set(offsets)) != len(offsets)
                or any(x not in OFFSETS for x in offsets)):
            raise ValueError("Missing evidence or invalid offsets")
        if packet["condition"] == "center" and (offsets != [0] or item["reference_comparability"] != "not_provided"):
            raise ValueError("Context leakage into center condition")
        if item["scene"] != "forward_or_aligned_road" and item["reference_grade"] != "abstain":
            raise ValueError("Out-of-scope forward grade")
    usage = finals[0].get("usage", {})
    return dict(items=items, image_attachments=len(packet["expected_images"]), conversation_id=session,
                offset_semantics="Raw model-reported offsets, may list reference-only images; center delivery independently logged; not proof of visual attention",
                usage={k: usage.get(v, 0) for k, v in {"input_tokens": "inputTokens", "output_tokens": "outputTokens", "thinking_tokens": "thoughtTokens", "total_tokens": "totalTokens"}.items()})


def run_one(packet, output, agy):
    output.mkdir()
    directory = Path(packet["cwd"])
    for name in ["prompt", "schema"]:
        if sha(directory / (name + "." + ("txt" if name == "prompt" else "json"))) != packet[name + "_sha256"]:
            raise ValueError("Frozen protocol changed")
    blocks = [{"type": "text", "text": (directory / "prompt.txt").read_text(encoding="utf-8")
               + "\nReturn JSON matching this schema exactly, no prose:\n" + (directory / "schema.json").read_text(encoding="utf-8")}]
    for filename, digest in packet["expected_images"].items():
        raw = Path(filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Image changed before attachment")
        blocks += [{"type": "text", "text": "Attached image: " + Path(filename).name},
                   {"type": "image", "mimeType": "image/jpeg", "data": base64.b64encode(raw).decode("ascii")}]
    settings = {"permissions": {"deny": ["command(*)", "write_file(*)"]}, "useG1Credits": False}
    cmd = [str(agy), "--settings", json.dumps(settings), "acp", "--disable-telemetry", "--include-default-customizations=false"]
    dump(output / "invocation.json", dict(args=cmd, cwd=str(directory), timeout_seconds=600, client_capabilities={}))
    started = time.monotonic()
    events, mailbox = [], queue.Queue()
    status = dict(valid=False)
    with (output / "stderr.txt").open("w", encoding="utf-8") as error_log, (output / "stdout.jsonl").open("w", encoding="utf-8") as transcript:
        proc = subprocess.Popen(cmd, cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error_log, text=True, encoding="utf-8")

        def reader():
            for line in proc.stdout:
                mailbox.put(line)
            mailbox.put(None)

        threading.Thread(target=reader, daemon=True).start()

        def record(event):
            events.append(event)
            transcript.write(json.dumps(event, ensure_ascii=False) + "\n")
            transcript.flush()

        def send(event):
            proc.stdin.write(json.dumps(event) + "\n")
            proc.stdin.flush()

        def request(identifier, method, params):
            send(dict(jsonrpc="2.0", id=identifier, method=method, params=params))
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                try:
                    line = mailbox.get(timeout=min(1, max(.01, deadline-time.monotonic())))
                except queue.Empty:
                    continue
                if line is None:
                    raise ValueError("ACP process closed output")
                event = json.loads(line)
                record(event)
                if "id" in event and "method" in event:
                    # A denied request is retained and invalidates the batch.
                    if event["method"] == "session/request_permission":
                        send(dict(jsonrpc="2.0", id=event["id"], result={"outcome": {"outcome": "cancelled"}}))
                    else:
                        send(dict(jsonrpc="2.0", id=event["id"], error={"code": -32601, "message": "Client capability not offered"}))
                if event.get("id") == identifier and ("result" in event or "error" in event):
                    if "error" in event:
                        raise ValueError("ACP error: " + str(event["error"]))
                    return event["result"]
            raise TimeoutError("ACP request timeout: " + method)

        try:
            init = request(0, "initialize", dict(protocolVersion=1, clientCapabilities={}, clientInfo=dict(name="blind-image-review", version="0.1")))
            if not init.get("agentCapabilities", {}).get("promptCapabilities", {}).get("image"):
                raise ValueError("ACP lacks image support")
            session = request(1, "session/new", dict(cwd=str(directory), mcpServers=[]))
            if session.get("models", {}).get("currentModelId") != MODEL:
                raise ValueError("Default model differs from frozen model; do not silently substitute")
            # No base64 data or secrets in this client delivery summary.
            record(dict(event="client_attachment_delivery", images=packet["expected_images"], prompt_sha256=packet["prompt_sha256"]))
            request(2, "session/prompt", dict(sessionId=session["sessionId"], prompt=blocks))
            result = validate_packet(packet, events)
            dump(output / "validated.json", result)
            status.update(valid=True, cases=len(result["items"]))
        except (ValueError, KeyError, TypeError, TimeoutError, OSError) as error:
            status["error"] = str(error)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    status["elapsed"] = time.monotonic()-started
    dump(output / "status.json", status)
    return status
