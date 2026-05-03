"""
Interactive chat with a local GGUF model via llama-cpp-python.
Type your prompt and press Enter. Type 'quit' to stop and save results JSON.

Usage:
  python scripts/interactive_chat.py
  python scripts/interactive_chat.py --model models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
from llama_cpp import Llama

DEFAULT_MODEL = "models/Llama-3.2-1B-Instruct-f16.gguf"
N_CTX      = 2048
N_THREADS  = 4
EXIT_WORD  = "quit"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model file")
args = parser.parse_args()
MODEL_PATH = Path(args.model)
RESULTS_DIR = Path("results")

# SYSTEM_PROMPT = "You are a helpful assistant running locally on a Raspberry Pi 5, answer in a concise manner."
SYSTEM_PROMPT = ""

def detect_family(p: Path) -> str:
    name = p.name.lower()
    if "qwen" in name:
        return "qwen"
    if "mistral" in name or "ministral" in name:
        return "mistral"
    return "llama3"


FAMILY = detect_family(MODEL_PATH)
STOP_TOKENS = {
    "llama3":  ["<|eot_id|>", "<|end_of_text|>"],
    "qwen":    ["<|im_end|>"],
    "mistral": ["</s>", "[INST]"],
}[FAMILY]


def build_prompt(history: list[dict]) -> str:
    if FAMILY == "qwen":
        out = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        for msg in history:
            out += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        out += "<|im_start|>assistant\n"
        return out
    if FAMILY == "mistral":
        # Mistral v1: system injected into first user turn
        turns = []
        for i, msg in enumerate(history):
            if msg["role"] == "user":
                content = msg["content"]
                if i == 0:
                    content = f"{SYSTEM_PROMPT}\n\n{content}"
                turns.append(f"[INST] {content} [/INST]")
            else:
                turns.append(msg["content"] + "</s>")
        return "<s>" + " ".join(turns)
    # llama3
    out = f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
    for msg in history:
        out += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content']}<|eot_id|>"
    out += "<|start_header_id|>assistant<|end_header_id|>\n"
    return out


class ResourceMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._running = False
        self._thread  = None
        self.cpu_samples: list[float] = []
        self.ram_samples: list[float] = []

    def start(self):
        self.cpu_samples.clear()
        self.ram_samples.clear()
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def _run(self):
        proc = psutil.Process(os.getpid())
        while self._running:
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.ram_samples.append(proc.memory_info().rss / 1024 / 1024)
            time.sleep(self.interval)

    def summary(self) -> dict:
        def safe_mean(lst): return round(sum(lst) / len(lst), 2) if lst else 0
        def safe_peak(lst): return round(max(lst), 2) if lst else 0
        return {
            "cpu_mean": safe_mean(self.cpu_samples),
            "cpu_peak": safe_peak(self.cpu_samples),
            "ram_mean_mb": safe_mean(self.ram_samples),
            "ram_peak_mb": safe_peak(self.ram_samples),
        }


def save_json(records: list[dict], model_name: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"chat_{model_name}_{ts}.json"
    payload = {
        "model":     model_name,
        "timestamp": datetime.now().isoformat(),
        "n_ctx":     N_CTX,
        "n_threads": N_THREADS,
        "turns":     records,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {out}")


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading {MODEL_PATH.name} ...")
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=0,
        verbose=False,
    )
    print(f"Ready in {time.perf_counter() - t0:.1f}s  |  type '{EXIT_WORD}' to stop\n")

    history: list[dict] = []
    records: list[dict] = []
    monitor = ResourceMonitor()
    turn    = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == EXIT_WORD:
            print("Bye!")
            break

        turn += 1
        history.append({"role": "user", "content": user_input})
        prompt = build_prompt(history)

        print("Assistant: ", end="", flush=True)

        monitor.start()
        t_start          = time.perf_counter()
        first_token_time = None
        reply            = ""
        token_count      = 0

        for chunk in llm(prompt, max_tokens=512, stream=True, echo=False,
                         stop=STOP_TOKENS):
            token = chunk["choices"][0]["text"]
            if first_token_time is None and token:
                first_token_time = time.perf_counter()
            print(token, end="", flush=True)
            reply       += token
            token_count += 1

        elapsed = time.perf_counter() - t_start
        monitor.stop()
        res = monitor.summary()

        ttft = round(first_token_time - t_start, 3) if first_token_time else None
        tps  = round(token_count / elapsed, 2) if elapsed > 0 else None

        print(f"\n  [{token_count} tokens | {tps} tok/s | TTFT {ttft}s]\n")

        history.append({"role": "assistant", "content": reply.strip()})
        records.append({
            "turn":             turn,
            "timestamp":        datetime.now().isoformat(),
            "user_input":       user_input,
            "assistant_output": reply.strip(),
            "output_tokens":    token_count,
            "ttft_s":           ttft,
            "total_time_s":     round(elapsed, 3),
            "tokens_per_sec":   tps,
            "cpu_mean_%":       res["cpu_mean"],
            "cpu_peak_%":       res["cpu_peak"],
            "ram_mean_mb":      res["ram_mean_mb"],
            "ram_peak_mb":      res["ram_peak_mb"],
        })

    if records:
        save_json(records, MODEL_PATH.stem)


if __name__ == "__main__":
    main()
