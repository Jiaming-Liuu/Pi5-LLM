"""
Efficiency benchmark: fixed prompts → tok/s, TTFT, load time, RAM, CPU, power.
Results saved to results/efficiency/efficiency_<model>_<timestamp>.json

Usage:
  python scripts/eval_efficiency.py
  python scripts/eval_efficiency.py --model models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
  python scripts/eval_efficiency.py --max-tokens 128
"""

import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
from llama_cpp import Llama

parser = argparse.ArgumentParser()
parser.add_argument("--model",      default="models/Llama-3.2-1B-Instruct-f16.gguf")
parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate per prompt")
args = parser.parse_args()

MODEL_PATH  = Path(args.model)
MAX_TOKENS  = args.max_tokens
N_CTX       = 2048
N_THREADS   = 4
RESULTS_DIR = Path("results/efficiency")

# Fixed prompts — short, medium, long input to stress different context lengths
PROMPTS = [
    # short inputs
    "What is 7 times 8?",
    "Name the capital of France.",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
    "What does CPU stand for?",
    # medium inputs
    "Explain what machine learning is in two sentences.",
    "Describe the water cycle briefly.",
    "What are the main differences between RAM and ROM?",
    "Summarize the French Revolution in three sentences.",
    "How does a neural network learn from data?",
    # longer inputs (force more prefill work)
    (
        "You are helping a student understand embedded systems. "
        "The student asks: what are the tradeoffs between running a large language model "
        "on a cloud server versus on a local device like a Raspberry Pi? "
        "Please give a concise answer covering latency, privacy, and cost."
    ),
    (
        "Given that a Raspberry Pi 5 has 8GB of RAM and a quad-core ARM Cortex-A76 "
        "running at 2.4GHz, estimate how many tokens per second a 3-billion parameter "
        "model quantized to 4-bit would achieve, and explain your reasoning."
    ),
]


class ResourceMonitor:
    def __init__(self, interval: float = 0.25):
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
        def mean(lst): return round(sum(lst) / len(lst), 2) if lst else 0.0
        def peak(lst): return round(max(lst), 2) if lst else 0.0
        return {
            "cpu_mean_%":    mean(self.cpu_samples),
            "cpu_peak_%":    peak(self.cpu_samples),
            "ram_mean_mb":   mean(self.ram_samples),
            "ram_peak_mb":   peak(self.ram_samples),
        }


def build_prompt(user_text: str) -> str:
    return (
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant running locally on a Raspberry Pi 5, answer concisely.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def try_read_power_w() -> float | None:
    """Read battery/power draw from UPS HAT via I2C if available."""
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        # INA219-style read — adjust register/address for your HAT
        raw = bus.read_word_data(0x40, 0x04)
        bus.close()
        raw = ((raw & 0xFF) << 8) | (raw >> 8)
        return round(raw * 0.001, 3)  # mW → W
    except Exception:
        return None


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_PATH.name} ...")
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=0,
        verbose=False,
    )
    load_time = round(time.perf_counter() - t0, 2)
    print(f"Loaded in {load_time}s\n")

    # Warm-up: one silent generation to prime the model
    llm("Hi", max_tokens=4, echo=False)

    monitor  = ResourceMonitor()
    runs     = []
    all_toks = []
    all_ttft = []

    print(f"Running {len(PROMPTS)} fixed prompts (max_tokens={MAX_TOKENS})\n")
    for idx, user_text in enumerate(PROMPTS):
        prompt      = build_prompt(user_text)
        first_token = True
        ttft        = None
        tokens_out  = 0
        t_start     = None
        response    = []

        monitor.start()
        power_before = try_read_power_w()

        t0 = time.perf_counter()

        # Stream so we can record TTFT
        for chunk in llm(
            prompt,
            max_tokens=MAX_TOKENS,
            echo=False,
            stream=True,
            stop=["<|eot_id|>"],
        ):
            if first_token:
                ttft = round(time.perf_counter() - t0, 3)
                t_start = time.perf_counter()
                first_token = False
            text = chunk["choices"][0].get("text", "")
            response.append(text)
            tokens_out += 1

        total_time  = round(time.perf_counter() - t0, 3)
        power_after = try_read_power_w()
        monitor.stop()

        gen_time = total_time - (ttft or 0)
        toks_per_s = round(tokens_out / gen_time, 2) if gen_time > 0 else None

        power_w = None
        if power_before is not None and power_after is not None:
            power_w = round((power_before + power_after) / 2, 3)

        resources = monitor.summary()

        rec = {
            "prompt_idx":   idx + 1,
            "prompt":       user_text,
            "response":     "".join(response).strip(),
            "tokens_out":   tokens_out,
            "ttft_s":       ttft,
            "gen_time_s":   round(gen_time, 3),
            "total_time_s": total_time,
            "toks_per_s":   toks_per_s,
            "power_w":      power_w,
            **resources,
        }
        runs.append(rec)
        if toks_per_s is not None: all_toks.append(toks_per_s)
        if ttft       is not None: all_ttft.append(ttft)

        print(
            f"  [{idx+1:02d}/{len(PROMPTS)}] "
            f"ttft={ttft:.2f}s  toks/s={toks_per_s or '?':>6}  "
            f"ram_peak={resources['ram_peak_mb']:.0f}MB  "
            f"cpu={resources['cpu_mean_%']:.0f}%"
        )

    def mean(lst): return round(sum(lst) / len(lst), 2) if lst else None

    summary = {
        "mean_toks_per_s": mean(all_toks),
        "mean_ttft_s":     mean(all_ttft),
        "min_toks_per_s":  round(min(all_toks), 2) if all_toks else None,
        "max_toks_per_s":  round(max(all_toks), 2) if all_toks else None,
    }

    print(f"\n{'='*52}")
    print(f"  EFFICIENCY SUMMARY — {MODEL_PATH.stem}")
    print(f"{'='*52}")
    print(f"  Load time      : {load_time}s")
    print(f"  Mean tok/s     : {summary['mean_toks_per_s']}")
    print(f"  Mean TTFT      : {summary['mean_ttft_s']}s")
    print(f"  Tok/s range    : {summary['min_toks_per_s']} – {summary['max_toks_per_s']}")
    print(f"{'='*52}")

    output = {
        "model":        MODEL_PATH.name,
        "timestamp":    datetime.now().isoformat(),
        "load_time_s":  load_time,
        "n_ctx":        N_CTX,
        "n_threads":    N_THREADS,
        "max_tokens":   MAX_TOKENS,
        "n_prompts":    len(PROMPTS),
        "summary":      summary,
        "runs":         runs,
    }

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"efficiency_{MODEL_PATH.stem}_{ts}.json"
    out.write_text(json.dumps(output, indent=2))
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
