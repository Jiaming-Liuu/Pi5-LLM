"""
Full benchmark suite: HellaSwag + MMLU + TruthfulQA + WikiText-2
Results saved to results/benchmarks/benchmark_<model>_<timestamp>.json

Usage:
  python experiments/eval_benchmark.py
  python experiments/eval_benchmark.py --model models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
  python experiments/eval_benchmark.py --model models/Llama-3.2-3B-Instruct-Q4_K_M.gguf --n 100
  python experiments/eval_benchmark.py --skip hellaswag mmlu   # skip specific datasets
"""

import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
from datasets import load_dataset
from llama_cpp import Llama

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--model",          default="models/Llama-3.2-1B-Instruct-f16.gguf")
parser.add_argument("--n",              type=int, default=100, help="Samples per dataset")
parser.add_argument("--seed",           type=int, default=42,  help="Random seed for dataset shuffling")
parser.add_argument("--skip",           nargs="*", default=[], help="Datasets to skip: hellaswag, mmlu, truthfulqa")
parser.add_argument("--save-incorrect", action="store_true",   help="Save only incorrect answers in output")
parser.add_argument("--save-all",       action="store_true",   help="Save every question in output")
parser.add_argument("--threads",        type=int, default=4,   help="CPU threads (default 4 for Pi5; use more for local)")
parser.add_argument("--gpu-layers",     type=int, default=0,   help="Layers offloaded to GPU (0 = CPU only)")
args = parser.parse_args()

MODEL_PATH      = Path(args.model)
N               = args.n
SEED            = args.seed
SAVE_INCORRECT  = args.save_incorrect
SAVE_ALL        = args.save_all
N_CTX       = 2048
N_THREADS   = args.threads


def detect_family(p: Path) -> str:
    name = p.name.lower()
    if "qwen" in name:
        return "qwen"
    if "mistral" in name or "ministral" in name:
        return "mistral"
    return "llama3"


FAMILY  = detect_family(MODEL_PATH)
MC_STOP = {"llama3": ["<|eot_id|>", "\n"],
            "qwen":   ["<|im_end|>",  "\n"],
            "mistral":["</s>",        "\n"]}[FAMILY]
RESULTS_DIR = Path("results/benchmarks")
CHOICES     = ["A", "B", "C", "D"]

MMLU_SUBJECTS = [
    "high_school_geography",
    "world_religions",
    "elementary_mathematics",
    "high_school_biology",
    "astronomy",
]  # 20 questions each = 100 total


# Resource monitor
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
        def mean(lst): return round(sum(lst) / len(lst), 2) if lst else 0
        def peak(lst): return round(max(lst), 2) if lst else 0
        return {
            "cpu_mean":   mean(self.cpu_samples),
            "cpu_peak":   peak(self.cpu_samples),
            "ram_mean_mb": mean(self.ram_samples),
            "ram_peak_mb": peak(self.ram_samples),
        }


# Shared helpers
def extract_letter(text: str, valid: list[str] | None = None) -> str | None:
    pool = valid if valid else CHOICES
    for ch in text.strip():
        if ch.upper() in pool:
            return ch.upper()
    return None


def log_mc(i: int, n: int, correct: int, is_correct: bool, t: float):
    acc    = round(correct / (i + 1) * 100, 1)
    status = "PASS" if is_correct else "FAIL"
    print(f"  [{i+1:03d}/{n}] {status}  acc={acc:5.1f}%  ({t:.1f}s)")


def keep_record(record: dict) -> bool:
    if SAVE_ALL:       return True
    if SAVE_INCORRECT: return not record.get("correct", True)
    return False


def mc_prompt(system: str, question_block: str) -> str:
    if FAMILY == "qwen":
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{question_block}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n"
        )
    if FAMILY == "mistral":
        return f"<s>[INST] {system}\n\n{question_block} [/INST]"
    return (
        f"<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{question_block}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


# 1. HellaSwag — zero-shot commonsense completion
def run_hellaswag(llm: Llama, n: int) -> dict:
    print("\nHellaSwag (zero-shot)")
    ds = load_dataset("Rowan/hellaswag", split="validation").shuffle(seed=SEED).select(range(n))
    monitor = ResourceMonitor()
    records, correct = [], 0

    for i, row in enumerate(ds):
        activity = row["activity_label"]
        context  = row["ctx"]
        endings  = row["endings"]
        answer   = CHOICES[int(row["label"])]
        options  = "\n".join(f"{CHOICES[j]}) {endings[j]}" for j in range(len(endings)))

        prompt = mc_prompt(
            "Choose the most plausible sentence ending. Reply with one letter only: A, B, C, or D.",
            f"Activity: {activity}\nContext: {context}\n\n{options}\n\nAnswer:"
        )

        monitor.start()
        t0  = time.perf_counter()
        out = llm(prompt, max_tokens=4, echo=False, stop=MC_STOP)
        t   = round(time.perf_counter() - t0, 3)
        monitor.stop()

        pred       = extract_letter(out["choices"][0]["text"])
        is_correct = pred == answer
        if is_correct:
            correct += 1

        rec = {
            "q": i + 1, "activity": activity, "context": context,
            "choices": {CHOICES[j]: endings[j] for j in range(4)},
            "answer": answer, "predicted": pred,
            "correct": is_correct, "time_s": t,
            "resources": monitor.summary(),
        }
        if keep_record(rec): records.append(rec)
        log_mc(i, n, correct, is_correct, t)

    acc = round(correct / n * 100, 1)
    print(f"  => HellaSwag accuracy: {acc}% ({correct}/{n})")
    return {"dataset": "hellaswag", "shot": "zero-shot",
            "accuracy_%": acc, "correct": correct, "n": n, "records": records}


# 2. MMLU — 5-shot multi-domain knowledge
def run_mmlu(llm: Llama, n: int) -> dict:
    print("\nMMLU (5-shot)")
    per_subject = n // len(MMLU_SUBJECTS)
    monitor     = ResourceMonitor()
    all_records, total_correct = [], 0

    for subject in MMLU_SUBJECTS:
        dev  = load_dataset("cais/mmlu", subject, split="dev")    # 5 few-shot examples
        test = load_dataset("cais/mmlu", subject, split="test").shuffle(seed=SEED).select(range(per_subject))

        # Build 5-shot prefix
        few_shot = ""
        for ex in dev:
            opts = "\n".join(f"{CHOICES[k]}) {ex['choices'][k]}" for k in range(4))
            few_shot += f"Q: {ex['question']}\n{opts}\nA: {CHOICES[ex['answer']]}\n\n"

        correct = 0
        for i, row in enumerate(test):
            opts   = "\n".join(f"{CHOICES[k]}) {row['choices'][k]}" for k in range(4))
            answer = CHOICES[row["answer"]]
            block  = (f"{few_shot}Q: {row['question']}\n{opts}\nA:")

            prompt = mc_prompt(
                f"Answer the following {subject.replace('_',' ')} questions. "
                "Reply with one letter only: A, B, C, or D.",
                block
            )

            monitor.start()
            t0  = time.perf_counter()
            out = llm(prompt, max_tokens=4, echo=False, stop=MC_STOP)
            t   = round(time.perf_counter() - t0, 3)
            monitor.stop()

            pred       = extract_letter(out["choices"][0]["text"])
            is_correct = pred == answer
            if is_correct:
                correct += 1
            total_correct += (1 if is_correct else 0)
            global_i = MMLU_SUBJECTS.index(subject) * per_subject + i

            rec = {
                "q": global_i + 1, "subject": subject, "question": row["question"],
                "choices": {CHOICES[k]: row["choices"][k] for k in range(4)},
                "answer": answer, "predicted": pred,
                "correct": is_correct, "time_s": t,
                "resources": monitor.summary(),
            }
            if keep_record(rec): all_records.append(rec)
            log_mc(global_i, n, total_correct, is_correct, t)

        subj_acc = round(correct / per_subject * 100, 1)
        print(f"  => {subject}: {subj_acc}%")

    acc = round(total_correct / n * 100, 1)
    print(f"  => MMLU overall accuracy: {acc}% ({total_correct}/{n})")
    return {"dataset": "mmlu", "shot": "5-shot", "subjects": MMLU_SUBJECTS,
            "accuracy_%": acc, "correct": total_correct, "n": n, "records": all_records}


# 3. TruthfulQA — zero-shot factual accuracy
def run_truthfulqa(llm: Llama, n: int) -> dict:
    print("\nTruthfulQA (zero-shot, mc1)")
    ds      = load_dataset("truthful_qa", "multiple_choice", split="validation").shuffle(seed=SEED).select(range(n))
    monitor = ResourceMonitor()
    records, correct = [], 0

    for i, row in enumerate(ds):
        question = row["question"]
        choices  = row["mc1_targets"]["choices"]
        labels   = row["mc1_targets"]["labels"]
        letters  = [chr(65 + k) for k in range(len(choices))]  # A, B, C... for any length
        try:
            answer = letters[labels.index(1)]
        except ValueError:
            continue  # malformed row — no correct label
        opts     = "\n".join(f"{letters[k]}) {choices[k]}" for k in range(len(choices)))
        valid    = "/".join(letters)

        prompt = mc_prompt(
            f"Answer the following question truthfully. Reply with one letter only: {valid}.",
            f"Q: {question}\n{opts}\nAnswer:"
        )

        monitor.start()
        t0  = time.perf_counter()
        out = llm(prompt, max_tokens=4, echo=False, stop=MC_STOP)
        t   = round(time.perf_counter() - t0, 3)
        monitor.stop()

        pred       = extract_letter(out["choices"][0]["text"], valid=letters)
        is_correct = pred == answer
        if is_correct:
            correct += 1

        rec = {
            "q": i + 1, "question": question,
            "choices": {letters[k]: choices[k] for k in range(len(choices))},
            "answer": answer, "predicted": pred,
            "correct": is_correct, "time_s": t,
            "resources": monitor.summary(),
        }
        if keep_record(rec): records.append(rec)
        log_mc(i, n, correct, is_correct, t)

    acc = round(correct / n * 100, 1)
    print(f"  => TruthfulQA accuracy: {acc}% ({correct}/{n})")
    return {"dataset": "truthfulqa", "shot": "zero-shot",
            "accuracy_%": acc, "correct": correct, "n": n, "records": records}



def main():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_PATH.name} ...")
    t0 = time.perf_counter()
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=N_CTX, n_threads=N_THREADS,
                n_gpu_layers=args.gpu_layers, logits_all=True, verbose=False)
    load_time = round(time.perf_counter() - t0, 2)
    print(f"Loaded in {load_time}s — running {N} samples per dataset\n")

    skip    = [s.lower() for s in args.skip]
    results = {
        "model":         MODEL_PATH.name,
        "model_family":  FAMILY,
        "timestamp":     datetime.now().isoformat(),
        "load_time_s":   load_time,
        "n_ctx":         N_CTX,
        "n_threads":     N_THREADS,
        "n_gpu_layers":  args.gpu_layers,
        "n_per_dataset": N,
        "seed":          SEED,
        "datasets":      {},
    }

    if "hellaswag"  not in skip: results["datasets"]["hellaswag"]  = run_hellaswag(llm, N)
    if "mmlu"       not in skip: results["datasets"]["mmlu"]       = run_mmlu(llm, N)
    if "truthfulqa" not in skip: results["datasets"]["truthfulqa"] = run_truthfulqa(llm, N)

    # Summary
    print("\n" + "="*54)
    print(f"  BENCHMARK SUMMARY — {MODEL_PATH.stem}")
    print("="*54)
    for name, r in results["datasets"].items():
        print(f"  {name:<12} accuracy   : {r['accuracy_%']}% ({r['correct']}/{r['n']})")
    print("="*54)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"benchmark_{MODEL_PATH.stem}_{ts}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()