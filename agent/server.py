"""Interactive agent loop for Qwen3.5-2B Q4_K_M with two skills.

Usage:
  python -m agent.server --model models/Qwen_Qwen3.5-2B-Q4_K_M.gguf
  python -m agent.server --model models/Qwen_Qwen3.5-2B-Q4_K_M.gguf --no-tools

Each turn:
  1. Constrained JSON tool call from the LLM (GBNF).
  2. Execute the chosen skill.
  3. Feed the result back to the LLM unconstrained for a natural-language reply.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .grammar import tool_call_grammar
from .llm import Agent
from .skills import execute, skill_descriptions

DEFAULT_MODEL = "/home/student/dev_james/Pi5-LLM/models/Qwen_Qwen3.5-2B-Q4_K_M.gguf"

PLANNER_SYSTEM = """You are an offline Pi 5 assistant.
Decide which skill to call NEXT to make progress on the user's request. You will be called multiple times; previous tool results appear in the conversation.

Available skills:
{skills}

When the user's request is fully addressed (all needed reads done, all needed writes done), reply:
{{"skill": "done", "args": {{}}}}

Otherwise reply with ONLY a JSON object:
{{"skill": "<skill_name>", "args": {{...}}}}

Use {{}} for args if the skill takes none. Do not write any text outside the JSON."""

REPLY_SYSTEM = """You are an offline Pi 5 assistant.
Answer the user in 2-4 sentences using the tool result below. Stick to the data provided; if the user asks for something the tool result cannot deliver (e.g. writing a file when no write skill exists), say so briefly."""


def stream_tokens(tokens) -> str:
    """Stream tokens to stdout and return the full concatenated text."""
    full = ""
    for token in tokens:
        full += token
        sys.stdout.write(token)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return full


def parse_tool_call(raw: str) -> tuple[str, dict] | None:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
        return obj["skill"], obj.get("args", {}) or {}
    except (json.JSONDecodeError, KeyError):
        return None


MAX_STEPS = 10


def turn(agent: Agent, user_msg: str, history: list[dict]) -> str:
    """Run one user turn. Tool-call limits and repeat counters are LOCAL —
    they reset every call, so a fresh request always gets a fresh budget.
    Multi-turn memory comes from `history`: the planner sees only the current
    request + in-turn tool results (so prior turns can't trigger early `done`),
    while the reply phase sees full history for natural continuity."""
    planner_sys = PLANNER_SYSTEM.format(skills=skill_descriptions())
    grammar = tool_call_grammar()

    # Planner's view: just this turn (prevents prior turns from being read as completed work).
    planner_view: list[dict] = [{"role": "user", "content": user_msg}]
    last_call: tuple[str, str] | None = None
    repeat_count = 0

    for step in range(MAX_STEPS):
        call = None
        for attempt in range(3):
            raw = agent.generate(
                system=planner_sys,
                history=planner_view,
                max_tokens=512,
                grammar=grammar,
                temperature=0.0 if attempt > 0 else 0.2,
            )
            call = parse_tool_call(raw)
            if call:
                break
        if not call:
            print("  [planner] parse failed, ending tool loop", file=sys.stderr)
            break

        skill_name, args = call
        if skill_name == "done":
            print(f"  [step {step + 1}] done", file=sys.stderr)
            break

        call_key = (skill_name, json.dumps(args, sort_keys=True))
        if call_key == last_call:
            repeat_count += 1
        else:
            last_call = call_key
            repeat_count = 1
        if repeat_count > 3:
            print(f"  [step {step + 1}] {skill_name}({args}) repeated 3x; breaking loop", file=sys.stderr)
            break

        print(f"  [step {step + 1}] {skill_name}({args})", file=sys.stderr)
        result = execute(skill_name, args)
        result_str = json.dumps(result, indent=2)
        planner_view = planner_view + [
            {"role": "assistant", "content": json.dumps({"skill": skill_name, "args": args})},
            {"role": "user", "content": f"[tool:{skill_name} result]\n{result_str}"},
        ]
    else:
        print(f"  [planner] hit MAX_STEPS={MAX_STEPS}", file=sys.stderr)

    # Reply phase: full conversation history + this turn's user msg + in-turn tool results.
    reply_view = history + planner_view
    print("Assistant: ", end="", flush=True)
    full = stream_tokens(
        agent.stream(
            system=REPLY_SYSTEM,
            history=reply_view,
            max_tokens=400,
            temperature=0.3,
        )
    )
    return full.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--n-threads", type=int, default=4)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {model_path.name}...", file=sys.stderr)
    t0 = time.perf_counter()
    agent = Agent(model_path, n_ctx=args.n_ctx, n_threads=args.n_threads)
    print(f"Ready in {time.perf_counter() - t0:.1f}s. Type 'quit' to exit.\n", file=sys.stderr)

    history: list[dict] = []
    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_msg:
            continue
        if user_msg.lower() in {"quit", "exit"}:
            break
        if user_msg.lower() == "reset":
            history = []
            print("  [history cleared]\n", file=sys.stderr)
            continue
        if user_msg.lower() == "history":
            print(f"  [{len(history)} messages in history]", file=sys.stderr)
            for m in history:
                print(f"    {m['role']}: {m['content'][:80]}", file=sys.stderr)
            print(file=sys.stderr)
            continue

        t = time.perf_counter()
        reply = turn(agent, user_msg, history)
        print(f"  [{time.perf_counter() - t:.1f}s]\n")
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})

        # rolling window — keep last 6 turns
        if len(history) > 12:
            history = history[-12:]


if __name__ == "__main__":
    main()
