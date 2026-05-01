# Pi5-LLM Agent

Offline LLM agent for the Raspberry Pi 5, powered by **Qwen3.5-2B Q4_K_M**.
Two skills: `read_status` (BME280 + UPS HAT + CPU) and `read_PDF` (sandboxed PDF text extraction).

## Layout

```
agent/
  llm.py        # Qwen ChatML prompt + llama-cpp wrapper
  grammar.py    # GBNF: forces {"skill": ..., "args": {...}}
  skills.py     # read_status, read_PDF, registry
  server.py     # CLI loop: plan -> tool -> reply
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place the model at:
mkdir -p models
cp ../../dev_james/Pi5-LLM/models/Qwen_Qwen3.5-2B-Q4_K_M.gguf models/
```

## Run

```bash
python -m agent.server --model models/Qwen_Qwen3.5-2B-Q4_K_M.gguf
```

PDFs are read from `$PI5_SANDBOX` (default `~/pi5_sandbox`).

## Loop

1. User message → planner LLM (GBNF-constrained) emits one tool-call JSON.
2. Skill executes, returns dict.
3. Result fed back to LLM unconstrained for natural-language reply.
4. Up to 3 attempts on parse failure before giving up.

## Notes

- `read_status` falls back to CPU-only stats if I2C hardware is absent (dev-machine friendly).
- `read_PDF` rejects paths that escape the sandbox and truncates output at ~700 tokens.
- History is trimmed to the last 12 messages to stay inside the 2048-token context.
