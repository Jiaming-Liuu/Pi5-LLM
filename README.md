# Pi5-LLM

## Benchmark Results (Part 1) — COMPLETE

### Accuracy

| Model | HellaSwag | MMLU | TruthfulQA |
|---|---|---|---|
| Llama-3.2-1B F16 | 35% | 19% | 31% |
| Llama-3.2-3B Q4_K_M | 34% | 22% | 36% |
| Llama-3.2-3B Q8_0 | 38% | 20% | 40% |
| Llama-3.2-3B F16 | 39% | 20% | 43% |
| Ministral-3B Q4_K_M | 36% | 25% | 30% |
| Ministral-3B Q8_0 | 44% | 29% | 26% |
| Ministral-3B BF16 | 43% | 31% | 24% |
| **Qwen3.5-2B Q4_K_M** | **45%** | **54%** | **59%** |
| **Qwen3.5-2B Q8_0** | **40%** | **59%** | **61%** |
| **Qwen3.5-2B BF16** | **39%** | **60%** | **57%** |
| Llama-3-8B Q4_K_M | 63% | 64% | 40% |

### Efficiency

| Model | tok/s | TTFT | RAM peak | Max Temp |
|---|---|---|---|---|
| Llama-3.2-1B F16 | 5.03 | 0.55s | 2558 MB | 62.8°C |
| Llama-3.2-3B Q4_K_M | 6.42 | 0.96s | 4198 MB | 64.5°C |
| Llama-3.2-3B Q8_0 | 3.59 | 1.65s | 6406 MB | 63.4°C |
| Llama-3.2-3B F16 | 1.70 | 2.23s | 6343 MB | 63.4°C |
| Ministral-3B Q4_K_M | 5.52 | 0.86s | 4412 MB | 64.5°C |
| Ministral-3B Q8_0 | 3.08 | 1.52s | 6430 MB | 61.7°C |
| Ministral-3B BF16 | 0.65 | 28.82s | 6718 MB | 65.5°C |
| **Qwen3.5-2B Q4_K_M** ⭐ | **7.11** | 1.00s | **2748 MB** | 65.0°C |
| Qwen3.5-2B Q8_0 | 4.79 | 1.42s | 4051 MB | 62.8°C |
| Qwen3.5-2B BF16 | 1.04 | 31.85s | 3836 MB | 65.5°C |
| Llama-3-8B Q4_K_M | 2.61 | 2.69s | 6816 MB | 65.0°C |
