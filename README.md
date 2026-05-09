# Pi5-LLM — TinyBrain

A fully offline, tool-using LLM agent that runs end-to-end on a single Raspberry Pi 5 (8 GB), powered by a battery UPS HAT. The agent uses a grammar-constrained planner / sandboxed executor / streaming reply loop, all backed by a quantized 2 B-parameter Qwen3.5 model running on `llama.cpp`. No cloud, no network calls, no vendor lock-in — once the model file is on disk, every prompt is served from the Pi.

This repository contains:

- The agent itself (planner, executor, five skills, GBNF grammar, repair logic)
- A Flask + SSE web UI with a dual-mode sandbox: files can live on the Pi, or on the user's PC via the browser's File System Access API
- A reproducible benchmark suite covering 11 model/quantization configurations across HellaSwag, MMLU, TruthfulQA, and a fixed-prompt latency harness
- The final report (4-page ACM sigconf format) and the demo deck

Final project for **Embedded AI**, Columbia University, Spring 2026.
Authors: Thomas Duan (`hd2593`), Jiaming Liu (`jl7250`).

---

## Highlights

- **Fully offline.** All inference, file I/O, and tool execution happens on-device. The web UI is served from the Pi over LAN.
- **Five skills.** `read_status` (CPU/RAM/temp + BME280 + UPS HAT), `list_dir`, `read_file`, `read_PDF`, `write_file`. New skills are a single registry entry away.
- **Grammar-constrained tool calls.** The planner emits exactly one JSON object per step, enforced by a GBNF grammar passed to `llama.cpp`. No JSON parsing surprises, no hallucinated tool names.
- **Dual-mode sandbox.** Run the agent against a folder on the Pi, or pick a folder on your PC at runtime — the agent's skills are the same; the Flask server forwards file ops over a server-sent-events bridge to the browser.
- **Robustness mechanisms.** Truncation-aware JSON repair, repeat-call detection, streamed-output loop detection, and Qwen3.5 `<think>...</think>` block stripping.
- **Reproducible benchmarks.** Single-command evaluation of any GGUF model on three datasets and a 12-prompt latency harness, with per-run resource sampling.

---

## Hardware

Required:

- Raspberry Pi 5, 8 GB RAM
- microSD card, 64 GB Class 10 (the model files alone are 1.3–6.8 GB)

Optional but supported by the `read_status` skill:

- Waveshare UPS HAT (D) with 4× 21700 Li-ion cells (battery monitoring over I²C)

The agent runs CPU-only. All numbers below are with `n_threads=4` on the four Cortex-A76 cores at 2.4 GHz.

---

## Installation

```bash
git clone https://github.com/<your-org>/Pi5-LLM.git
cd Pi5-LLM

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`llama-cpp-python` will compile from source on first install; on the Pi 5 this takes 5–10 minutes. If you don't have the I²C sensors connected, the install still succeeds — `read_status` will fall back to CPU/RAM only.

### Downloading models

All commands write into `models/`. See [`model_download.md`](model_download.md) for the full list of variants. The minimum needed to run the agent:

```bash
pip install -U "huggingface_hub[cli]"
hf download bartowski/Qwen_Qwen3.5-2B-GGUF \
    --include "Qwen_Qwen3.5-2B-Q4_K_M.gguf" \
    --local-dir models/
```

This downloads roughly 1.3 GB.

---

## Quick Start

### Web UI (recommended)

```bash
python -m interface.app --model models/Qwen_Qwen3.5-2B-Q4_K_M.gguf
```

Open `http://<pi-ip>:8000` from any device on the same network. On first load you'll be asked to choose a sandbox:

- **Use Pi sandbox** — files live in `~/pi5_sandbox/` on the Pi. Drag-and-drop upload, click-to-download, in-browser preview for `.txt`, `.py`, and `.pdf`.
- **Select PC folder** — pick a directory on your laptop (Chrome/Edge only). The agent on the Pi calls back into your browser to read and write that folder, with no SMB share or copying.

### CLI agent (no UI)

```bash
python -m agent.server --model models/Qwen_Qwen3.5-2B-Q4_K_M.gguf
```

Type a request, the agent will plan and execute tool calls in the terminal, then stream a reply. Useful for SSH-only sessions and for debugging the planner.

### Plain chat (no tools)

For benchmark-style chat with token throughput and TTFT printed after each turn:

```bash
python scripts/interactive_chat.py --model models/Qwen_Qwen3.5-2B-Q4_K_M.gguf
```

---

## Architecture

```
                     +------- loop --------+
                     v                     |
   User -----> Planner ------ call ------> Executor
                |                          (5 skills)
                | done
                v
              Reply -------- stream -------> User

   (Planner and Reply share one Qwen3.5-2B Q4_K_M instance)
```

**Planner.** A GBNF grammar restricts the model to emit exactly one JSON object of the form `{"skill": <name>, "args": {...}}`. The grammar is defined in `agent/grammar.py`. The planner's view is restricted to the current turn's tool exchanges only — prior assistant replies in the conversation history are hidden from it, otherwise the planner reads "an answer already exists" and emits `done` immediately.

**Executor.** Dispatches the parsed call to one of the five skills in `agent/skills.py`. The result is JSON-serialized and appended to the planner's view. The loop terminates when the planner emits `{"skill": "done"}`, when the same `(skill, args)` pair repeats more than three times, or when the 10-step cap is hit.

**Reply.** Same model, no grammar, full conversation history. Tokens stream to the user; a small state machine drops `<think>...</think>` blocks Qwen3.5 sometimes emits.

---

## Skills

| Skill         | Args              | Description                                                  |
| ------------- | ----------------- | ------------------------------------------------------------ |
| `read_status` | none              | Returns CPU temp/load/RAM, BME280 environment, and UPS HAT battery state. Fails gracefully without I²C hardware. |
| `list_dir`    | `path` (optional) | Lists files and subdirectories in the active sandbox.        |
| `read_file`   | `path`            | Reads UTF-8 text files (`.txt`, `.md`, `.py`, `.json`, ...). Truncates to ~12 KB. |
| `read_PDF`    | `path`            | Extracts text from a PDF via `pypdf`. Truncates to ~12 KB and reports `truncated: true`. |
| `write_file`  | `path`, `content` | Writes a UTF-8 text file inside the sandbox. Overwrites if it exists. |

Add a new skill by registering it in the `REGISTRY` dict in `agent/skills.py` and adding a new alternative to the `skill` rule in `agent/grammar.py`.

---

## Sandbox Modes

The agent never touches the filesystem directly — every skill goes through a `Sandbox` instance defined in `connection/sandbox.py`. Two backends are provided:

**`LocalSandbox`** — paths resolve under `~/pi5_sandbox/` on the Pi (override with `PI5_SANDBOX=/some/path`). Path traversal is blocked; absolute paths are normalized to their basename so an LLM hallucinating `/home/pi/foo.pdf` still resolves to `foo.pdf` in the sandbox.

**`BrowserSandbox`** — proxies every operation to the user's browser through a single-consumer SSE bridge (`connection/bridge.py`). The browser holds a long-running connection to `/fs/poll`; the Pi-side skill calls `bridge.request(op, args)` which generates a UUID, queues the op, and blocks on a `threading.Event` with a 30 s timeout. The browser executes the op against a `FileSystemDirectoryHandle` from `window.showDirectoryPicker()` and POSTs the result back to `/fs/result/<id>`, which sets the event and wakes the Pi-side thread.

PC mode requires Chrome or Edge (the File System Access API is not available in Firefox or Safari) and a secure context — either HTTPS, or the Chromium flag at `chrome://flags/#unsafely-treat-insecure-origin-as-secure` with the Pi's origin allowlisted.

---

## Reproducing the Benchmarks

The project ships two harnesses, both in `scripts/`. All results land in `results/benchmarks/` and `results/efficiency/` as timestamped JSON.

### Accuracy: HellaSwag, MMLU, TruthfulQA

```bash
python scripts/eval_benchmark.py \
    --model models/Qwen_Qwen3.5-2B-Q4_K_M.gguf \
    --n 100
```

100 samples per dataset, MMLU 5-shot across 5 subjects (high school geography, world religions, elementary mathematics, high school biology, astronomy), HellaSwag 0-shot, TruthfulQA MC1 0-shot. Datasets are loaded via Hugging Face `datasets`; the first run downloads ~600 MB of cached parquet files.

Pass `--skip mmlu truthfulqa` to run a subset, `--save-incorrect` to keep only failed questions in the JSON output, or `--save-all` to keep every question.

### Efficiency: tok/s, TTFT, RAM, temperature

```bash
python scripts/eval_efficiency.py \
    --model models/Qwen_Qwen3.5-2B-Q4_K_M.gguf
```

12 fixed prompts with `max_tokens=128`, sampled at 4 Hz for CPU and RAM, with CPU temperature read from `/sys/class/thermal/thermal_zone0/temp` before and after each prompt. Power draw is read from the UPS HAT INA219 if present.

### Running the full sweep

To reproduce Table 1 from the report:

```bash
for m in models/*.gguf; do
    python scripts/eval_benchmark.py --model "$m" --n 100
    python scripts/eval_efficiency.py --model "$m"
done
```

On the Pi 5 this takes roughly 12 hours wall-clock (the BF16 variants are slow).

---

## Benchmark Results

100 samples per dataset; HellaSwag 0-shot, MMLU 5-shot, TruthfulQA MC1 0-shot. Accuracy in percent.

### Accuracy

| Model                 | HellaSwag |   MMLU | TruthfulQA |
| --------------------- | --------: | -----: | ---------: |
| Llama-3.2-1B F16      |        35 |     19 |         31 |
| Llama-3.2-3B Q4_K_M   |        34 |     22 |         36 |
| Llama-3.2-3B Q8_0     |        38 |     20 |         40 |
| Llama-3.2-3B F16      |        39 |     20 |         43 |
| Ministral-3B Q4_K_M   |        36 |     25 |         30 |
| Ministral-3B Q8_0     |        44 |     29 |         26 |
| Ministral-3B BF16     |        43 |     31 |         24 |
| **Qwen3.5-2B Q4_K_M** |    **45** | **54** |     **59** |
| Qwen3.5-2B Q8_0       |        40 |     59 |         61 |
| Qwen3.5-2B BF16       |        39 |     60 |         57 |
| Llama-3-8B Q4_K_M     |        63 |     64 |         40 |

### Efficiency

| Model                 |    tok/s | TTFT (s) | RAM peak (MB) | Max temp (°C) |
| --------------------- | -------: | -------: | ------------: | ------------: |
| Llama-3.2-1B F16      |     5.03 |     0.55 |          2558 |          62.8 |
| Llama-3.2-3B Q4_K_M   |     6.42 |     0.96 |          4198 |          64.5 |
| Llama-3.2-3B Q8_0     |     3.59 |     1.65 |          6406 |          63.4 |
| Llama-3.2-3B F16      |     1.70 |     2.23 |          6343 |          63.4 |
| Ministral-3B Q4_K_M   |     5.52 |     0.86 |          4412 |          64.5 |
| Ministral-3B Q8_0     |     3.08 |     1.52 |          6430 |          61.7 |
| Ministral-3B BF16     |     0.65 |    28.82 |          6718 |          65.5 |
| **Qwen3.5-2B Q4_K_M** | **7.11** | **1.00** |      **2748** |          65.0 |
| Qwen3.5-2B Q8_0       |     4.79 |     1.42 |          4051 |          62.8 |
| Qwen3.5-2B BF16       |     1.04 |    31.85 |          3836 |          65.5 |
| Llama-3-8B Q4_K_M     |     2.61 |     2.69 |          6816 |          65.0 |

Qwen3.5-2B Q4_K_M is the only configuration that simultaneously achieves the best throughput in our set, sub-second TTFT, and the top TruthfulQA score among ≤3 B models, while leaving roughly 5 GB of headroom on the 8 GB Pi for the OS and the agent runtime. It is the model deployed in the agent.

---

## End-to-End Agent Evaluation

A LeetCode code-generation task: the agent reads a problem statement from a file, writes a Python solution to a new file, and confirms. End-to-end on the Pi 5, Qwen3.5-2B Q4_K_M.

|    # | Problem                           | Difficulty | Result  | Latency (s) |
| ---: | --------------------------------- | ---------- | ------- | ----------: |
|    1 | Two Sum                           | Easy       | Pass    |         186 |
|    2 | Valid Parentheses                 | Easy       | Pass    |         130 |
|    3 | Best Time to Buy/Sell Stock II    | Medium     | Pass    |         115 |
|    4 | Binary Tree Level Order Traversal | Medium     | Pass    |          67 |
|    5 | Trapping Rain Water               | Hard       | Fail    |         139 |
|      | **Mean / total**                  |            | **4/5** |     **127** |

The single failure on Trapping Rain Water was *not* an algorithmic error — the model produced a correct two-pointer solution. The submission threw `IndexError: list index out of range` because the model omitted the empty-input edge case, and the LeetCode judge probes empty input first. A single `if not height: return 0` would have flipped the result to Pass. We trace this to our planner system prompt, which explicitly instructs the model to omit edge-case guards because including them tended to overflow the JSON `content` field and break the tool-call grammar.

---

## Project Structure

```
Pi5-LLM/
├── agent/
│   ├── grammar.py        GBNF grammar for tool-call JSON
│   ├── llm.py            llama.cpp wrapper with Qwen ChatML prompt builder
│   ├── server.py         CLI agent loop (planner -> executor -> reply)
│   └── skills.py         Skill registry: read_status, list_dir, read_file, read_PDF, write_file
├── connection/
│   ├── bridge.py         Single-consumer SSE bridge for browser RPC
│   └── sandbox.py        LocalSandbox + BrowserSandbox backends
├── interface/
│   ├── app.py            Flask + SSE web server, tool-call streaming, PC sandbox bridge
│   └── templates/
│       └── index.html    Single-page UI: chat, sandbox sidebar, mode picker
├── scripts/
│   ├── eval_benchmark.py     Accuracy: HellaSwag / MMLU / TruthfulQA
│   ├── eval_efficiency.py    Latency, RAM, CPU, temperature
│   └── interactive_chat.py   Plain chat without tools
├── results/
│   ├── benchmarks/       JSON output of eval_benchmark.py
│   └── efficiency/       JSON output of eval_efficiency.py
├── models/               Place GGUF model files here (gitignored)
├── model_download.md     hf download commands for every variant we tested
├── requirements.txt
├── LICENSE               MIT
└── README.md
```

---

## License

MIT, see [LICENSE](LICENSE).

