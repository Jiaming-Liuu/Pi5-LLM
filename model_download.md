# Model Download Commands

All models download to `models/`. Run from the project root with the venv active.

---

## Llama-3.2-1B-Instruct

| Quant | Size | Status |
|-------|------|--------|
| f16   | ~2.9 GB | ✓ downloaded |

```bash
hf download bartowski/Llama-3.2-1B-Instruct-GGUF \
  --include "Llama-3.2-1B-Instruct-f16.gguf" \
  --local-dir models/
```

---

## Qwen3.5-2B

Source: `bartowski/Qwen_Qwen3.5-2B-GGUF`

| Quant   | Size    | Status |
|---------|---------|--------|
| Q4_K_M  | 1.33 GB | ✓ downloaded |
| Q8_0    | 2.02 GB | ✓ downloaded |
| bf16    | 3.78 GB | ✓ downloaded |

```bash
# Q4_K_M
hf download bartowski/Qwen_Qwen3.5-2B-GGUF \
  --include "Qwen_Qwen3.5-2B-Q4_K_M.gguf" \
  --local-dir models/

# Q8_0
hf download bartowski/Qwen_Qwen3.5-2B-GGUF \
  --include "Qwen_Qwen3.5-2B-Q8_0.gguf" \
  --local-dir models/

# bf16
hf download bartowski/Qwen_Qwen3.5-2B-GGUF \
  --include "Qwen_Qwen3.5-2B-bf16.gguf" \
  --local-dir models/
```

---

## Llama-3.2-3B-Instruct

| Quant   | Size    | Status |
|---------|---------|--------|
| Q4_K_M  | ~2.0 GB | pending |
| Q8_0    | ~3.4 GB | pending |
| f16     | ~6.4 GB | pending |

```bash
# Q4_K_M
hf download bartowski/Llama-3.2-3B-Instruct-GGUF \
  --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  --local-dir models/

# Q8_0
hf download bartowski/Llama-3.2-3B-Instruct-GGUF \
  --include "Llama-3.2-3B-Instruct-Q8_0.gguf" \
  --local-dir models/

# f16
hf download bartowski/Llama-3.2-3B-Instruct-GGUF \
  --include "Llama-3.2-3B-Instruct-f16.gguf" \
  --local-dir models/
```

---

## Ministral-3B

Source: `mistralai/Ministral-3-3B-Instruct-2512-GGUF` (official)

| Quant   | Size    | Status |
|---------|---------|--------|
| Q4_K_M  | 2.15 GB | pending |
| Q8_0    | 3.65 GB | pending |
| bf16    | 6.87 GB | pending |

```bash
# Q4_K_M
hf download mistralai/Ministral-3-3B-Instruct-2512-GGUF \
  --include "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf" \
  --local-dir models/

# Q8_0
hf download mistralai/Ministral-3-3B-Instruct-2512-GGUF \
  --include "Ministral-3-3B-Instruct-2512-Q8_0.gguf" \
  --local-dir models/

# bf16
hf download mistralai/Ministral-3-3B-Instruct-2512-GGUF \
  --include "Ministral-3-3B-Instruct-2512-BF16.gguf" \
  --local-dir models/
```

---

## Llama-3-8B-Instruct

| Quant   | Size    | Status |
|---------|---------|--------|
| Q4_K_M  | 4.7 GB  | pending |
| Q8_0    | 8.5 GB  | pending |

```bash
# Q4_K_M
hf download bartowski/Meta-Llama-3-8B-Instruct-GGUF \
  --include "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" \
  --local-dir models/

# Q8_0
hf download bartowski/Meta-Llama-3-8B-Instruct-GGUF \
  --include "Meta-Llama-3-8B-Instruct-Q8_0.gguf" \
  --local-dir models/
```