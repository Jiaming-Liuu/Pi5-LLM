"""Flask web UI for the Pi5 agent.

Run from the project root:
    python -m interface.app --model models/Qwen_Qwen3.5-2B-Q4_K_M.gguf

Then open http://<pi-ip>:8000 from your PC.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory, stream_with_context
from werkzeug.utils import secure_filename

from agent.grammar import tool_call_grammar
from agent.llm import Agent
from agent.server import MAX_STEPS, PLANNER_SYSTEM, REPLY_SYSTEM, parse_tool_call
from agent.skills import SANDBOX_ROOT, execute, get_sandbox, set_sandbox, skill_descriptions
from connection.bridge import Bridge
from connection.sandbox import BrowserSandbox, LocalSandbox

DEFAULT_MODEL = "/home/student/dev_james/Pi5-LLM/models/Qwen_Qwen3.5-2B-Q4_K_M.gguf"
HISTORY_LIMIT = 12
PLANNER_MAX_TOKENS = 2048
PLANNER_REPEAT_PENALTY = 1.15
LOOP_TAIL_LEN = 60        # tail substring length to look for repetition
LOOP_TAIL_THRESHOLD = 4   # how many times that substring may appear before we call it a loop

app = Flask(__name__)
state: dict = {
    "agent": None,
    "history": [],
    "lock": threading.Lock(),
    "bridge": Bridge(),
    "mode": None,   # None | "pi" | "pc"
}


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def sse_response(generator):
    return Response(stream_with_context(generator), mimetype="text/event-stream", headers=SSE_HEADERS)


def log(msg: str) -> None:
    """Timestamped debug line to stderr (visible in the terminal running the server)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def preview(text, n: int = 140) -> str:
    s = str(text).replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def redact_args(skill_name: str, args: dict) -> dict:
    """Hide write_file content from logs/UI; keep length so progress is visible."""
    if skill_name != "write_file" or "content" not in args:
        return args
    out = dict(args)
    out["content"] = f"<{len(args['content'])} chars elided>"
    return out


def looks_like_loop(text: str, tail_len: int = LOOP_TAIL_LEN, threshold: int = LOOP_TAIL_THRESHOLD) -> bool:
    """Detect if the model is stuck repeating: count how many times the last
    `tail_len` chars appear within the recent buffer. If it shows up `threshold`+
    times, we're in a loop."""
    if len(text) < tail_len * threshold:
        return False
    tail = text[-tail_len:]
    # count overlapping is overkill; non-overlapping is fine for catching loops
    return text.count(tail) >= threshold


def redact_partial(text: str) -> str:
    """Strip the value of any "content": "..." from a partial/complete JSON string.
    Used for the streaming planner tail when a write_file call is being generated.
    """
    if '"content"' not in text:
        return text
    idx = text.find('"content"')
    after = text.find(":", idx)
    if after < 0:
        return text[:idx] + '"content": "<elided>"'
    # Find opening quote of the content value
    q1 = text.find('"', after)
    if q1 < 0:
        return text[: after + 1] + ' "<elided>"'
    return text[: q1 + 1] + "<elided>"


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def strip_think(tokens):
    """Drop <think>...</think> blocks from a token stream, holding back partial tags."""
    buf = ""
    in_think = False
    for token in tokens:
        buf += token
        out = ""
        while True:
            if not in_think:
                idx = buf.find(THINK_OPEN)
                if idx >= 0:
                    out += buf[:idx]
                    buf = buf[idx + len(THINK_OPEN):]
                    in_think = True
                    continue
                hold = len(THINK_OPEN) - 1
                if len(buf) > hold:
                    out += buf[:-hold]
                    buf = buf[-hold:]
                break
            else:
                idx = buf.find(THINK_CLOSE)
                if idx >= 0:
                    buf = buf[idx + len(THINK_CLOSE):]
                    in_think = False
                    continue
                hold = len(THINK_CLOSE) - 1
                if len(buf) > hold:
                    buf = buf[-hold:]
                break
        if out:
            yield out
    if not in_think and buf:
        yield buf


def run_turn(user_msg: str):
    """Generator yielding SSE chunks for one user turn."""
    agent: Agent = state["agent"]
    grammar = tool_call_grammar()
    planner_sys = PLANNER_SYSTEM.format(skills=skill_descriptions())

    log(f"=== TURN START | user: {preview(user_msg)}")
    log(f"    history={len(state['history'])} msgs")
    turn_t0 = time.perf_counter()

    planner_view: list[dict] = [{"role": "user", "content": user_msg}]
    last_call: tuple[str, str] | None = None
    repeat_count = 0

    for step in range(MAX_STEPS):
        call = None
        truncated = False
        looped = False
        for attempt in range(3):
            log(f"[planner] step {step + 1}/{MAX_STEPS} attempt {attempt + 1}/3 | view={len(planner_view)} msgs | streaming (max_tokens={PLANNER_MAX_TOKENS}, repeat_penalty={PLANNER_REPEAT_PENALTY})…")
            t0 = time.perf_counter()
            chunks: list[str] = []
            n_tok = 0
            last_log = t0
            looped = False
            yield sse("planner_start", {"step": step + 1, "attempt": attempt + 1})
            for chunk in agent.stream(
                system=planner_sys,
                history=planner_view,
                max_tokens=PLANNER_MAX_TOKENS,
                grammar=grammar,
                temperature=0.0 if attempt > 0 else 0.2,
                repeat_penalty=PLANNER_REPEAT_PENALTY,
            ):
                chunks.append(chunk)
                n_tok += 1
                now = time.perf_counter()
                if now - last_log >= 1.5:
                    partial = "".join(chunks)
                    elapsed = now - t0
                    rate = n_tok / elapsed if elapsed > 0 else 0
                    log(f"[planner]   …{n_tok} chunks, {len(partial)} chars in {elapsed:.1f}s ({rate:.1f}/s) | tail: {preview(partial[-100:])}")
                    yield sse("planner_progress", {
                        "step": step + 1,
                        "chunks": n_tok,
                        "chars": len(partial),
                        "elapsed": round(elapsed, 1),
                        "tail": partial[-80:],
                    })
                    if looks_like_loop(partial):
                        log(f"[planner] degenerate loop detected (tail repeats ≥{LOOP_TAIL_THRESHOLD}× in output); aborting stream")
                        looped = True
                        break
                    last_log = now
            raw = "".join(chunks)
            dt = time.perf_counter() - t0
            log(f"[planner] step {step + 1} attempt {attempt + 1} done in {dt:.1f}s | {n_tok} chunks, {len(raw)} chars | raw: {preview(raw)}")
            call = parse_tool_call(raw)
            if call:
                break
            # Grammar-constrained output that fails to parse means it was truncated (max_tokens
            # or our loop detector). Retries at temp=0 reproduce the same output, so abort.
            truncated = n_tok >= PLANNER_MAX_TOKENS - 2
            if truncated or looped:
                why = "loop detected" if looped else f"hit max_tokens={PLANNER_MAX_TOKENS}"
                log(f"[planner] {why}; output unparseable | aborting retries")
                break
            log(f"[planner] parse failed on attempt {attempt + 1}; retrying")
        if not call:
            if looped:
                reason = "planner stuck in repetition loop"
            elif truncated:
                reason = f"output truncated at max_tokens={PLANNER_MAX_TOKENS}"
            else:
                reason = "planner parse failed"
            log(f"[planner] {reason}; ending tool loop")
            yield sse("error", {"message": reason})
            break

        skill_name, args = call
        if skill_name == "done":
            log(f"[planner] step {step + 1} -> DONE")
            break

        call_key = (skill_name, json.dumps(args, sort_keys=True))
        if call_key == last_call:
            repeat_count += 1
        else:
            last_call = call_key
            repeat_count = 1
        if repeat_count > 3:
            log(f"[planner] {skill_name}({redact_args(skill_name, args)}) repeated 3x; breaking loop")
            yield sse("error", {"message": f"{skill_name} repeated 3x"})
            break

        safe = redact_args(skill_name, args)
        log(f"[tool] -> {skill_name}({safe})")
        yield sse("tool", {"name": skill_name, "args": safe})
        t0 = time.perf_counter()
        result = execute(skill_name, args)
        dt = time.perf_counter() - t0
        result_str = json.dumps(result, indent=2)
        log(f"[tool] {skill_name} done in {dt:.1f}s | result: {preview(result_str, 200)}")
        yield sse("tool_result", {"name": skill_name, "result": result})
        planner_view = planner_view + [
            {"role": "assistant", "content": json.dumps({"skill": skill_name, "args": args})},
            {"role": "user", "content": f"[tool:{skill_name} result]\n{result_str}"},
        ]
    else:
        log(f"[planner] hit MAX_STEPS={MAX_STEPS}")

    reply_view = state["history"] + planner_view
    log(f"[reply] starting stream | view={len(reply_view)} msgs (history={len(state['history'])} + turn={len(planner_view)})")
    reply_t0 = time.perf_counter()
    full = ""
    n_chunks = 0
    last_log = reply_t0
    raw_stream = agent.stream(
        system=REPLY_SYSTEM,
        history=reply_view,
        max_tokens=800,
        temperature=0.3,
    )
    for token in strip_think(raw_stream):
        full += token
        n_chunks += 1
        yield sse("token", {"text": token})
        now = time.perf_counter()
        if now - last_log >= 2.0:
            elapsed = now - reply_t0
            rate = n_chunks / elapsed if elapsed > 0 else 0
            log(f"[reply] streaming… {n_chunks} chunks, {len(full)} chars in {elapsed:.1f}s ({rate:.1f} ch/s)")
            last_log = now

    reply_dt = time.perf_counter() - reply_t0
    rate = n_chunks / reply_dt if reply_dt > 0 else 0
    log(f"[reply] done | {n_chunks} chunks, {len(full)} chars in {reply_dt:.1f}s ({rate:.1f} ch/s)")

    state["history"].append({"role": "user", "content": user_msg})
    state["history"].append({"role": "assistant", "content": full.strip()})
    if len(state["history"]) > HISTORY_LIMIT:
        state["history"] = state["history"][-HISTORY_LIMIT:]
    elapsed = time.perf_counter() - turn_t0
    log(f"=== TURN DONE in {elapsed:.1f}s | history={len(state['history'])} msgs")
    yield sse("done", {"elapsed": round(elapsed, 1)})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "empty message"}), 400

    def stream():
        with state["lock"]:
            yield from run_turn(msg)

    return sse_response(stream())


@app.route("/reset", methods=["POST"])
def reset():
    with state["lock"]:
        state["history"] = []
    return jsonify({"ok": True})


@app.route("/history", methods=["GET"])
def history():
    return jsonify({"history": state["history"]})


@app.route("/sandbox/mode", methods=["GET", "POST"])
def sandbox_mode():
    if request.method == "GET":
        return jsonify({
            "mode": state["mode"],
            "label": get_sandbox().label if state["mode"] else None,
        })
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    with state["lock"]:
        if mode == "pi":
            set_sandbox(LocalSandbox(SANDBOX_ROOT))
            state["mode"] = "pi"
        elif mode == "pc":
            name = (data.get("name") or "PC folder")[:80]
            set_sandbox(BrowserSandbox(state["bridge"], name))
            state["mode"] = "pc"
        else:
            return jsonify({"error": "invalid mode"}), 400
        state["history"] = []   # confirmed: clear chat on mode switch
    log(f"[sandbox] mode={state['mode']} | {get_sandbox().label}")
    return jsonify({"ok": True, "mode": state["mode"], "label": get_sandbox().label})


@app.route("/sandbox/list", methods=["GET"])
def sandbox_list():
    if state["mode"] is None:
        return jsonify({"dir": "(no sandbox selected)", "entries": []})
    try:
        return jsonify(get_sandbox().list_dir(""))
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 200


@app.route("/sandbox/upload", methods=["POST"])
def sandbox_upload():
    """Pi-mode only. PC mode writes via the directory handle client-side."""
    if state["mode"] != "pi":
        return jsonify({"error": "upload only available in Pi mode"}), 400
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "no files"}), 400
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    saved, errors = [], []
    for f in files:
        if not f.filename:
            continue
        name = secure_filename(f.filename)
        if not name:
            errors.append({"name": f.filename, "error": "invalid filename"})
            continue
        target = SANDBOX_ROOT / name
        f.save(str(target))
        saved.append({"name": name, "bytes": target.stat().st_size})
        log(f"[upload] saved {name} ({target.stat().st_size} bytes) -> {target}")
    return jsonify({"saved": saved, "errors": errors})


@app.route("/sandbox/download/<path:p>", methods=["GET"])
def sandbox_download(p):
    """Pi-mode only. PC mode downloads directly from the directory handle in JS."""
    if state["mode"] != "pi":
        return ("download only available in Pi mode", 404)
    return send_from_directory(str(SANDBOX_ROOT), p, as_attachment=True)


@app.route("/sandbox/preview/<path:p>", methods=["GET"])
def sandbox_preview(p):
    """Pi-mode only. Serves the file inline so the browser can render it natively
    (PDFs in the built-in viewer; .txt/.py as plain text)."""
    if state["mode"] != "pi":
        return ("preview only available in Pi mode", 404)
    return send_from_directory(str(SANDBOX_ROOT), p, as_attachment=False)


@app.route("/sandbox/delete", methods=["POST"])
def sandbox_delete():
    """Pi-mode only. PC mode removes via the directory handle in JS."""
    if state["mode"] != "pi":
        return jsonify({"error": "delete only available in Pi mode"}), 400
    data = request.get_json(silent=True) or {}
    path = (data.get("path") or "").strip()
    if not path:
        return jsonify({"error": "path required"}), 400
    sb = get_sandbox()
    try:
        result = sb.delete(path)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400
    if "error" in result:
        return jsonify(result), 400
    log(f"[sandbox] deleted {path}")
    return jsonify(result)


# ---------- bridge endpoints (PC mode) ----------

@app.route("/fs/poll")
def fs_poll():
    bridge = state["bridge"]
    if not bridge.try_connect():
        return ("already connected from another tab", 409)
    log("[bridge] consumer connected")

    def gen():
        try:
            yield sse("hello", {})
            tick = 0
            while True:
                op = bridge.next_op(block_timeout=3.0)
                if op is None:
                    tick += 1
                    yield sse("ping", {"t": tick})
                    continue
                op_id, name, args = op
                yield sse("fs_op", {"id": op_id, "op": name, "args": args})
        finally:
            log("[bridge] consumer disconnected")
            bridge.disconnect()

    return sse_response(gen())


@app.route("/fs/result/<op_id>", methods=["POST"])
def fs_result(op_id):
    state["bridge"].complete(op_id, request.get_json(silent=True) or {})
    return ("", 204)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    print(f"Loading {model_path.name}...")
    t0 = time.perf_counter()
    state["agent"] = Agent(model_path, n_ctx=args.n_ctx, n_threads=args.n_threads)
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Ready in {time.perf_counter() - t0:.1f}s.")
    print(f"Sandbox: {SANDBOX_ROOT}")
    print(f"Listening on http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
